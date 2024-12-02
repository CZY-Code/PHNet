import os
import yaml
import json
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import cv2
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
]
ROOT = '../dataset'

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def generate_pred(info, lanes, img_num):
    pred_txt_path = './evaluation/txt/pred_txt'
    clip_name = info['name']
    imgName = info['ImgName'][img_num]
    size = info['size']
    if not os.path.exists(os.path.join(pred_txt_path, clip_name)):
        os.makedirs(os.path.join(pred_txt_path, clip_name))
    out_file_txt_name = imgName + '.lines.txt'
    with open(os.path.join(pred_txt_path, clip_name, out_file_txt_name), "w") as fp:
        for lane in lanes: #循环n条线
            if len(lane.points) > 2:
                for tx, ty in reversed(lane.points): #反转与否对指标没有影响
                    fp.write('%d %d ' % (tx * size[1], ty * size[0]))
                fp.write('\n')

def generate_predV2(info, lanes, img_num):
    pred_txt_path = './evaluation/txt4OL/pred_txt'
    clip_name = info['name']
    imgName = info['ImgName'][img_num]
    size = info['size']
    if not os.path.exists(os.path.join(pred_txt_path, clip_name)):
        os.makedirs(os.path.join(pred_txt_path, clip_name))
    out_file_txt_name = imgName + '.lines.txt'
    with open(os.path.join(pred_txt_path, clip_name, out_file_txt_name), "w") as fp:
        for lane in lanes: #循环n条线
            if len(lane.points) > 2:
                for tx, ty in reversed(lane.points): #反转与否对指标没有影响
                    # tx = tx / (1920 - 1) * (1920 // 2 - 1)
                    # ty = ty / (1280 - 1) * (1280 // 2 - 1)
                    fp.write('%.1f %.1f ' % (tx * size[1] / 2, (ty * size[0] + 480) / 2))
                fp.write('\n')


def generate_anno():
    #从标注的Json_org生成延伸到图像最下端的anno txt label
    anno_txt_path = './txt/anno_txt'
    data_dir = os.path.join(ROOT, 'VIL100')
    dbfile = os.path.join(data_dir, 'data', 'db_info.yaml')
    jsonpath = os.path.join(data_dir, 'Json')
    imgdir = os.path.join(data_dir, 'JPEGImages')
    train = False
    view = False
    # extract annotation information
    with open(dbfile, 'r') as f:
        db = yaml.load(f, Loader=yaml.Loader)['sequences']
        targetset = 'train' if train else 'test'
        videos = [info['name'] for info in db if info['set'] == targetset]
    for vid in videos: #循环每一个clip
        jsonfolder = os.path.join(jsonpath, vid)
        frames = [name[:5] for name in os.listdir(jsonfolder)]
        frames.sort()
        if not os.path.exists(os.path.join(anno_txt_path, vid)):
            os.makedirs(os.path.join(anno_txt_path, vid))

        for name in frames: #循环一帧
            print(jsonfolder, name+ '.jpg' + '.json')
            img = cv2.imread(os.path.join(imgdir, vid, name+'.jpg'))
            with open(os.path.join(jsonfolder, name+ '.jpg' + '.json')) as f:
                lanes_info = json.load(f)
                img_size = (lanes_info['info']['height'], lanes_info['info']['width'])
                lanes_anno = []
                for lane in lanes_info['annotations']['lane']:
                    lanes_anno.append(lane['points'])

            out_file_txt_name = name+'.lines.txt'
            with open(os.path.join(anno_txt_path, vid, out_file_txt_name), "w") as fp:
                for lane in lanes_anno: #循环n条线
                    # lane = sample_lane(lane, img_size)
                    if len(lane) <= 2: #过滤掉点数<=2的车道线
                        continue
                    for tx, ty in lane: #反转与否对指标没有影响
                        fp.write('%.1f %.1f ' % (tx, ty))
                    fp.write('\n')

                    if view:
                        xys = []
                        for x, y in lane:
                            if x <= 0 or y <= 0:
                                continue
                            x, y = int(x), int(y)
                            xys.append((x, y))
                            for i in range(1, len(xys)):
                                cv2.line(img, xys[i - 1], xys[i], COLORS[1], thickness=4)
            if view:
                cv2.imshow('pred_lane', img)
                cv2.waitKey(0)


def generate_json(view=False):
    data_dir = os.path.join(ROOT, 'VIL100')
    dbfile = os.path.join(data_dir, 'data', 'db_info.yaml')
    jsonpath = os.path.join(data_dir, 'Json')
    imgdir = os.path.join(data_dir, 'JPEGImages')
    train = False
    # extract annotation information
    with open(dbfile, 'r') as f:
        db = yaml.load(f, Loader=yaml.Loader)['sequences']
        targetset = 'train' if train else 'test'
        videos = [info['name'] for info in db if info['set'] == targetset]
        
    for vid in videos: #循环每一个clip
        jsonfolder = os.path.join(jsonpath, vid)
        frames = [name[:5] for name in os.listdir(jsonfolder)]
        frames.sort()

        for name in frames: #循环一帧
            print(jsonfolder, name+ '.jpg' + '.json')
            img = cv2.imread(os.path.join(imgdir, vid, name+'.jpg'))
            with open(os.path.join(jsonfolder, name+ '.jpg' + '.json'), 'r', encoding='utf-8') as jsonFile:
                lanes_info = json.load(jsonFile)
                img_size = (lanes_info['info']['height'], lanes_info['info']['width'])
                # for idx, lane in enumerate(lanes_info['annotations']['lane']):
                for idx in range(len(lanes_info['annotations']['lane'])-1, -1, -1): #只能倒序循环
                    points = lanes_info['annotations']['lane'][idx]['points']
                    points = sample_lane(points, img_size) #这里应该是in-place操作
                    lanes_info['annotations']['lane'][idx]['points'] = points #感觉有点多余
                    #删除点数小于等于2的线
                    if len(points) <= 2:
                        del lanes_info['annotations']['lane'][idx]
                        continue
                    if view:
                        xys = []
                        for x, y in points:
                            if x <= 0 or y <= 0:
                                continue
                            x, y = int(x), int(y)
                            xys.append((x, y))
                            for i in range(1, len(xys)):
                                cv2.line(img, xys[i - 1], xys[i], COLORS[1], thickness=4)
            if view:
                cv2.imshow('pred_lane', img)
                cv2.waitKey(0)

            with open(os.path.join(jsonfolder, name+ '.jpg' + '.json'), 'w', encoding='utf-8') as jsonFile:
                json.dump(lanes_info, jsonFile, ensure_ascii=False, cls=NpEncoder)


def filter_lane(lane):
    assert len(lane) >= 2, print("this is lane: ", lane)
    assert lane[-1][1] <= lane[0][1]
    filtered_lane = []
    used = set()
    for p in lane:
        if p[1] not in used:
            filtered_lane.append(p)
            used.add(p[1])
    return filtered_lane

def sample_lane(old_lane, img_size):
    img_h, img_w = img_size
    num_points = img_h//20 #10 #还存在一定的问题
    n_strips = num_points - 1
    strip_size = img_h / n_strips
    sample_ys = np.arange(img_h, -1, -strip_size)
    # removing points with less than 2 coordinate
    old_lane = filter(lambda x: len(x) > 1, old_lane)
    # sort lane points by Y (bottom to top of the image)
    old_lane = sorted(old_lane, key=lambda x: -x[1])
    # remove points with same Y (keep first occurrence)
    old_lane = filter_lane(old_lane)

    # this function expects the points to be sorted
    points = np.array(old_lane)
    if not np.all(points[1:, 1] < points[:-1, 1]):
        print(points)
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
    #remove the lane which cannot sample at least one point
    if len(sample_ys_inside_domain) < 2:
        print(domain_min_y, domain_max_y)
        return []
    assert len(sample_ys_inside_domain) >= 2, print(domain_min_y, domain_max_y)
    interp_xs = interp(sample_ys_inside_domain)

    # 底部车道线终止点到图像底线的部分
    # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
    two_closest_points = points[:2]
    extrap = np.polyfit(two_closest_points[:, 1],
                        two_closest_points[:, 0],
                        deg=1)
    extrap_ys = sample_ys[sample_ys > domain_max_y]
    extrap_xs = np.polyval(extrap, extrap_ys)
    all_xs = np.hstack((extrap_xs, interp_xs))
    all_yx = sample_ys[sample_ys>=domain_min_y]
    
    # all_xs = interp_xs
    # all_yx = sample_ys_inside_domain

    lane = np.concatenate((all_xs.reshape(-1, 1), all_yx.reshape(-1, 1)), axis=1)
    mask = (lane[:, 0] >= 0) & (lane[:, 0] < img_w)
    lane = lane[mask]
    return lane

if __name__ == '__main__':
    generate_json()
    generate_anno()