import numpy as np
import math
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import interpolate

from libs.dataset.openlane.utils import *

class Transforms(object):
    def __init__(self, cfg, training):
        self.cfg = cfg
        self.training = training
        self.settings()

    def settings(self):
        if self.cfg.mode_transform == 'basic':
            transforms = self.basic_transforms(self.cfg.height, self.cfg.width)
        elif self.cfg.mode_transform == 'complex':
            transforms = self.complex_transforms(self.cfg.height, self.cfg.width)
        else:
            transforms = self.custom_transforms(self.cfg.height, self.cfg.width)

        transforms_for_test = self.transforms_for_test(self.cfg.height, self.cfg.width)

        img_transforms = []
        for aug in transforms:
            p = aug['p']
            if aug['name'] != 'OneOf':
                img_transforms.append(
                    iaa.Sometimes(p=p, then_list=getattr(iaa, aug['name'])(**aug['parameters'])))
            else:
                img_transforms.append(
                    iaa.Sometimes(p=p, then_list=iaa.OneOf([getattr(iaa, aug_['name'])(**aug_['parameters']) for aug_ in aug['transforms']])))

        img_transforms_for_test = []
        for aug in transforms_for_test:
            p = aug['p']
            img_transforms_for_test.append(
                iaa.Sometimes(p=p, then_list=getattr(iaa, aug['name'])(**aug['parameters'])))

        self.transform = iaa.Sequential(img_transforms)
        self.transform_for_test = iaa.Sequential(img_transforms_for_test)

    def lane_to_linestrings(self, data):
        lane = list()
        for i in range(len(data)):
            pts = data[i]
            lane.append(LineString(pts))
        return lane

    def linestrings_to_lanes(self, data):
        lanes = []
        for pts in data:
            lanes.append(pts.coords)
        return lanes

    def process(self, img_org, anno):
        img_org = np.uint8(img_org)
        line_strings_org = self.lane_to_linestrings(anno)
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)
        # XXX
        for i in range(30):
            if self.training:
                img_new, line_strings = self.transform(image=img_org, line_strings=line_strings_org)
            else:
                img_new, line_strings = self.transform_for_test(image=img_org, line_strings=line_strings_org)
            lanes = self.linestrings_to_lanes(line_strings.clip_out_of_image_()) #XXX 是否需要clip_out_of_image_？
            try:
                anno_new = self.transform_annotation(self.cfg.height, self.cfg.width, lanes)
                break
            except Exception as e:
                if (i + 1) == 30:
                    self.logger.critical('Transform annotation failed 30 times :(')
                    exit()

        return img_new, anno_new

    def process_for_test(self, img_org, anno):
        img_org = np.uint8(img_org)
        line_strings_org = self.lane_to_linestrings(anno)
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)
        img_new, line_strings = self.transform_for_test(image=img_org, line_strings=line_strings_org)
        anno_new = self.linestrings_to_lanes(line_strings.clip_out_of_image_())
        return img_new, anno_new

    def check_one_to_one_mapping(self, data):
        dy = (data[:, 1][1:] - data[:, 1][:-1])
        c1 = np.sum(dy > 0)
        c2 = np.sum(dy <= 0)

        if c1 * c2 != 0:
            self.is_error_case['one_to_one'] = True
            # print(f'error case: not one-to-one mapping! {self.img_name}')

    def init_error_case(self, img_name):
        self.img_name = img_name
        self.is_error_case = dict()
        self.is_error_case['one_to_one'] = False
        self.is_error_case['fitting'] = False
        self.is_error_case['short'] = False
        self.is_error_case['iou'] = False
        self.is_error_case['total'] = False

    def get_lane_components(self, lanes):
        out = list()
        for i in range(len(lanes)):
            lane_pts = lanes[i]
            self.is_error_case['one_to_one'] = False
            self.is_error_case['fitting'] = False

            # check
            self.check_one_to_one_mapping(lane_pts)

            # remove duplicate pts
            unique_idx = np.sort(np.unique(lane_pts[:, 1], return_index=True)[1])
            lane_pts = lane_pts[unique_idx]
            unique_idx = np.sort(np.unique(lane_pts[:, 0], return_index=True)[1])
            lane_pts = lane_pts[unique_idx]

            # interpolation & extrapolation
            try:
                new_lane_pts = self.interp_extrap(lane_pts)
            except:
                self.is_error_case['fitting'] = True

            if self.is_error_case['one_to_one'] + self.is_error_case['fitting'] == 0:
                out.append(new_lane_pts[:, 0]) #取点的x轴坐标
            else:
                self.is_error_case['total'] = True
                break
        return {'extended_lanes': out}

    def interp_extrap(self, lane_pts):
        if lane_pts[0, 1] > lane_pts[-1, 1]:
            lane_pts = np.flip(lane_pts, axis=0)
        if self.cfg.mode_interp == 'spline':
            f = interpolate.InterpolatedUnivariateSpline(lane_pts[:, 1], lane_pts[:, 0], k=1)
            new_x_pts = f(self.cfg.py_coord)
        elif self.cfg.mode_interp == 'splrep':
            f = interpolate.splrep(lane_pts[:, 1], lane_pts[:, 0], k=1, s=5)
            new_x_pts = interpolate.splev(self.cfg.py_coord, f)
        else:
            f = interpolate.interp1d(lane_pts[:, 1], lane_pts[:, 0], kind=self.cfg.mode_interp, fill_value='extrapolate')
            new_x_pts = f(self.cfg.py_coord)

        new_lane_pts = np.concatenate((new_x_pts.reshape(-1, 1), self.cfg.py_coord.reshape(-1, 1)), axis=1)
        return new_lane_pts

    ### Option
    def custom_transforms(self, img_h, img_w):
        transform = [
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            # dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            # dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            # dict(name='MultiplyAndAddToBrightness',
            #      parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
            #      p=0.6),
            # dict(name='AddToHueAndSaturation',
            #      parameters=dict(value=(-10, 10)),
            #      p=0.7),
            # dict(name='OneOf',
            #      transforms=[
            #          dict(name='MotionBlur', parameters=dict(k=(3, 5))),
            #          dict(name='MedianBlur', parameters=dict(k=(3, 5)))
            #      ],
            #      p=0.2),
            # dict(name='Affine',
            #      parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
            #                                             y=(-0.1, 0.1)),
            #                      rotate=(-10, 10),
            #                      scale=(0.8, 1.2)),
            #      p=0.7),
            # dict(name='Resize',
            #      parameters=dict(size=dict(height=img_h, width=img_w)),
            #      p=1.0),
        ]
        return transform

    def transforms_for_test(self, img_h, img_w):
        transform = [
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ]
        return transform

    def basic_transforms(self, img_h, img_w):
        transform = [
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ]
        return transform

    def complex_transforms(self, img_h, img_w):
        transform = [
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.1), #p=0.5
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1), #p=0.5
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.9, 1.1), add=(-10, 10)), p=0.1), #p=0.5
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.1), #p=0.5
            dict(name='OneOf', #
                   transforms = [
                      dict(name='EdgeDetect', parameters=dict(alpha=(0, 0.2))),
                      dict(name='DirectedEdgeDetect', parameters=dict(alpha=(0, 0.2), direction=(0.0, 1.0)))
                   ],
                   p=0.1), #p=0.2
            dict(name='OneOf', 
                   transforms=[
                       dict(name='Dropout', parameters=dict(p=(0.0, 0.05), per_channel=0.5)),
                       dict(name='CoarseDropout', parameters=dict(p=(0.0, 0.05), size_percent=(0.02, 0.25), per_channel=0.2))
                   ],
                   p=0.1), #p=0.2
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5))),
                     dict(name='GaussianBlur', parameters=dict(sigma=(0, 3.0))) #
                 ],
                 p=0.1), #p=0.5
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                rotate=(-5, 5), #(-5,5)
                                scale=(0.9, 1.1)), #(0.9, 1.1)
                p=0.1),
            dict(name='Resize',
                parameters=dict(size=dict(height=img_h, width=img_w)),
                p=1.0),
        ]
        return transform

    def transform_annotation(self, img_h, img_w, anno):
        old_lanes = anno
        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 2, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        old_lanes = [[[
            x * img_w / float(img_w), y * img_h / float(img_h)
        ] for x, y in lane] for lane in old_lanes]
        # create tranformed annotations
        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, S+1 coordinates
        lanes = np.ones((self.cfg.max_lane_num, 2 + 1 + 1 + 2 + self.cfg.n_offsets), dtype=np.float32) * -1e5  
        
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.cfg.max_lane_num:
                break
            try:
                xs_outside_image, xs_inside_image = self.sample_lane(lane, self.cfg.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) <= 1:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.cfg.n_strips #starty
            lanes[lane_idx, 3] = xs_inside_image[0] / (img_w-1) #startx orgin:xs_inside_image[0]

            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = math.atan(i * self.cfg.strip_size /
                    (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / math.pi
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)
            theta_far = sum(thetas) / len(thetas)
            
            # lanes[lane_idx,
            #       4] = (theta_closest + theta_far) / 2  # averaged angle
            lanes[lane_idx, 4] = theta_far
            lanes[lane_idx, 5] = len(xs_inside_image)/self.cfg.n_strips #length
            lanes[lane_idx, 6:6 + len(all_xs)] = all_xs #实际位置 0～img_w
            # lanes[lane_idx, 6:6 + len(xs_inside_image)] = xs_inside_image #实际位置 0～img_w

        new_anno = {
            'label': lanes,
            'old_anno': anno,
        }
        return new_anno

    def sample_lane(self, points, sample_ys): #sample_ys:np.arange(self.height, -1, -self.strip_size)
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1], x[::-1],
                                              k=min(3, len(points) - 1))
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
        inside_mask = (all_xs >= 0) & (all_xs < self.cfg.width) #在y属于0-h范围内 x超出0-w的点
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
    