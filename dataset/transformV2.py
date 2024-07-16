import numpy as np
import torch
import math
import cv2
import copy
from PIL import Image
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from scipy.interpolate import InterpolatedUnivariateSpline
from .transform import Normalize

COLORS = [
    (0, 0, 0),
    (255, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128,),
    (128, 128, 128),
    (64, 0, 0),
    (191, 0, 0),
    (64, 128, 0),
    (191, 128, 0),
    (64, 0, 128),
    (191, 0, 128),
    (64, 128, 128),
    (191, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 191, 0),
    (128, 191, 0),
    (0, 64, 128)
]

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.
    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """
    def __init__(self, keys=['img', 'lane_line', 'seg', 'flow'], cfg=None):
        self.keys = keys

    def __call__(self, sample):
        data = dict()
        if len(sample['img'].shape) < 3:
            sample['img'] = np.expand_dims(sample['img'], -1) #维度扩展
        for key in self.keys:
            if key == 'img_metas' or key == 'gt_masks':
                data[key] = sample[key]
                continue
            data[key] = to_tensor(sample[key])
        data['img'] = data['img'].permute(2, 0, 1)
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


class GenerateLaneLine(object):
    def __init__(self, transforms=None, cfg=None, training=True):
        self.transforms = transforms
        self.img_h, self.img_w = cfg.img_h, cfg.img_w
        self.num_points = cfg.num_points
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        self.training = training
        self.normalize = Normalize()
        self.toTensor = ToTensor()
        
        if transforms is not None:
            img_transforms = []
            for aug in transforms:
                p = aug['p']
                if aug['name'] != 'OneOf':
                    img_transforms.append(
                        iaa.Sometimes(p=p,
                                      then_list=getattr(
                                          iaa,
                                          aug['name'])(**aug['parameters'])))
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf([
                                getattr(iaa,
                                        aug_['name'])(**aug_['parameters'])
                                for aug_ in aug['transforms']
                            ])))
        else:
            img_transforms = []
        self.transform = iaa.Sequential(img_transforms)
    
    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))
        return lines
    
    def sample_lane(self, points, sample_ys):
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1],
                                              x[::-1],
                                              k=min(3,
                                                    len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y)
                                            & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1],
                            two_closest_points[:, 0],
                            deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y] #底部车道线终止点到图像底线的部分
        extrap_xs = np.polyval(extrap, extrap_ys)

        all_xs = np.hstack((extrap_xs, interp_xs))
        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w) #在y属于0-h范围内 x超出0-w的点
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]
        return xs_outside_image, xs_inside_image
    
    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])
        return filtered_lane

    def transform_annotation(self, anno):
        img_w, img_h = self.img_w, self.img_h

        old_lanes = anno['lanes']

        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        old_lanes = [[[
            x * self.img_w / float(img_w), y * self.img_h / float(img_h)
        ] for x, y in lane] for lane in old_lanes]
        # create tranformed annotations
        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, S+1 coordinates
        lanes = np.ones((self.max_lanes, 2 + 1 + 1 + 2 + self.n_offsets), dtype=np.float32) * -1e5  
        lanes_endpoints = np.ones((self.max_lanes, 2))
        lanes_startpoints = np.zeros((self.max_lanes, 2))
        
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in zip(anno['lanes_ids'], old_lanes):
            if lane_idx >= self.max_lanes:
                break
            try:
                xs_outside_image, xs_inside_image = self.sample_lane(lane, self.offsets_ys)
                start_x, start_y = lane[0]
            except AssertionError:
                continue
            if len(xs_inside_image) <= 1:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips #starty
            lanes[lane_idx, 3] = xs_inside_image[0] / self.img_w #startx orgin:xs_inside_image[0]
            # lanes[lane_idx, 2] = (self.img_h-start_y)/self.img_h  #starty 从下到上起始点所在的比例
            # lanes[lane_idx, 3] = start_x #start_x x坐标

            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = math.atan(
                    i * self.strip_size /
                    (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / math.pi
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)

            theta_far = sum(thetas) / len(thetas)

            # lanes[lane_idx,
            #       4] = (theta_closest + theta_far) / 2  # averaged angle
            lanes[lane_idx, 4] = theta_far
            lanes[lane_idx, 5] = len(xs_inside_image)/self.n_strips #orgin: len(xs_inside_image)
            lanes[lane_idx, 6:6 + len(all_xs)] = all_xs #实际位置 0～640
            lanes_endpoints[lane_idx, 0] = self.img_h - (len(all_xs) - 1) * self.strip_size
            lanes_endpoints[lane_idx, 1] = xs_inside_image[-1]
            lanes_startpoints[lane_idx, 0] = self.img_h - len(xs_outside_image) * self.strip_size
            lanes_startpoints[lane_idx, 1] = xs_inside_image[0]

        new_anno = {
            'label': lanes,
            'old_anno': anno,
            'lane_endpoints': lanes_endpoints,
            'lane_startpoints': lanes_startpoints
        }
        return new_anno
    
    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)
        return lanes

    def __call__(self, sample):
        img_org = sample['img']

        line_strings_org = self.lane_to_linestrings(sample['lanes'])
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)
        mask_org = SegmentationMapsOnImage(sample['mask'], shape=img_org.shape)
        if sample['flow'] is not None:
            flow_org = HeatmapsOnImage(sample['flow'], shape=img_org.shape, min_value=-1, max_value=1)
        else:
            flow_org = None #FIXME

        for i in range(30):
            img, line_strings, seg, flow = self.transform(
                image=img_org.copy().astype(np.uint8),
                line_strings=line_strings_org,
                segmentation_maps=mask_org,
                heatmaps=flow_org)

            line_strings.clip_out_of_image_()
            new_anno = {'lanes': self.linestrings_to_lanes(line_strings),
                        'lanes_ids': sample['lanes_ids']}
            try:
                annos = self.transform_annotation(new_anno)
                label = annos['label']
                lane_endpoints = annos['lane_endpoints']
                lane_startpoints = annos['lane_startpoints']
                break
            except Exception as e:
                print(e)
                if (i + 1) == 30:
                    self.logger.critical('Transform annotation failed 30 times :(')
                    exit()
        
        # gt_vp_hm = np.zeros((1, 320, 800), np.float32) #没用
        # for endpoint in lane_endpoints:
        #     draw_umich_gaussian(gt_vp_hm[0], endpoint, radius=10)
        # for startpoint in lane_startpoints:
        #     draw_umich_gaussian(gt_vp_hm[0], startpoint, radius=10)
        # print(lane_endpoints)
        # cv2.imshow('img', img)
        # cv2.imshow('vp', gt_vp_hm[0])
        # cv2.waitKey(0)

        # sample['img'] = img.astype(np.float32) / 255. #暴力变成小数，没有使用归一化的方式
        sample['img'], _ = self.normalize(img.astype(np.float32), annos=None, use_image=None)
        sample['lane_line'] = label #72+6个点用于回归
        sample['lanes_endpoints'] = lane_endpoints
        sample['gt_points'] = new_anno['lanes'] #将图缩小到[320, 640]后的点list(array)

        sample['seg'] = seg.get_arr().astype(np.float32) #可能存在问题

        if sample['flow'] is not None:
            sample['flow'] = flow.get_arr().astype(np.float32) #可能存在问题
        
        # visualization(sample)
        sample = self.toTensor(sample)
        sample['seg'] = sample['seg'].permute(2,0,1).contiguous() #只包含[0, 1]的[9, H, W]
        # sample['flow'] #[320, 640, 2]
        return sample


def visualization(sample: dict):
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3]).astype(np.float32)
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3]).astype(np.float32)
    resized_img = copy.deepcopy(sample['img'])
    resized_img = ((resized_img * std + mean) * 255.0).astype(np.uint8)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    img_h, img_w = resized_img.shape[:-1]
    #----------------------seg--------------
    rescale_mask = cv2.resize(sample['seg'],(img_w, img_h))
    rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8) #(320, 640)
    seg_show = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for k in range(1, rescale_mask.max()+1):
        seg_show[rescale_mask==k, :] = COLORS[k-1] #sample['palette'][(k*3):(k+1)*3]
    im = cv2.addWeighted(resized_img, 0.5, seg_show, 0.5, 0, dtype = -1)
    cv2.imshow('img_seg', im)
    #-------------lane--------------------
    imshow_lanes(resized_img.copy(), sample['lanes_ids'], sample['gt_points'])
    #------org_img------------------
    cv2.imshow('img', resized_img)
    #-----flow----------
    flow = copy.deepcopy(sample['flow'])
    flow[..., 0] *= img_w #x-W [-1~1]
    flow[..., 1] *= img_h #y-H [-1~1]
    hsv = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    # 编码:将算法的输出转换为极坐标
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # 使用色相和饱和度来编码光流
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # 转换HSV图像为BGR
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow("frame", frame_copy)
    cv2.imshow("optical flow", bgr)

    cv2.waitKey(0)

def imshow_lanes(img, ids, lanes_gt, show=True, width=4):
    lanes_xys_gt = []
    for _, lane in enumerate(lanes_gt):
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append((x, y))
        lanes_xys_gt.append(xys)
    for idx, xys in zip(ids, lanes_xys_gt):
        for i in range(1, len(xys)):
            cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)
    if show:
        cv2.imshow('img_lane', img)
