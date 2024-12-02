import torch
import os
import cv2
import numpy as np
import torch
import json
import yaml
import random
from PIL import Image
from torch.utils.data import Dataset

from optionsV2 import OPTION as opt
from libs.utils.calcFlow import dense_twoFrame_flow

DATA_CONTAINER = {}
ROOT = opt.root
MAX_OBJECT = opt.max_object #车道线分割实例数目
MAX_TRAINING_SKIP = 100


def multibatch_collate_fn(batch):
    min_time = min([len(sample[0]) for sample in batch]) #一个cilp中包含的帧数
    frames = torch.stack([torch.stack([data['img'] for data in sample[0]]) for sample in batch])
    masks = torch.stack([torch.stack([data['seg'] for data in sample[0]]) for sample in batch])
    lanes_lines = torch.stack([torch.stack([data['lane_line'] for data in sample[0]]) for sample in batch])

    #------flow--------
    flows = torch.stack([torch.stack([data['flow'] for data in sample[0]]) for sample in batch])

    objs = [torch.LongTensor([sample[1]]) for sample in batch]
    objs = torch.cat(objs, dim=0)

    try:
        info = [sample[2] for sample in batch]
    except IndexError as ie:
        info = None
    return frames, masks, lanes_lines, objs, info, flows

def convert_mask(mask, max_obj):
    # convert mask to one hot encoded
    oh = []
    for k in range(max_obj+1):
        oh.append(mask==k)
    oh = np.stack(oh, axis=2)
    return oh

def convert_one_hot(oh, max_obj):
    mask = np.zeros(oh.shape[:2], dtype=np.uint8)
    for k in range(max_obj+1):
        mask[oh[:, :, k]==1] = k
    return mask

class BaseData(Dataset):
    def increase_max_skip(self):
        pass
    def set_max_skip(self):
        pass

class VIL(BaseData):
    def __init__(self, train=True, sampled_frames=3,
                 transform=None, max_skip=5, 
                 increment=5, samples_per_video=10,
                 calc_flow=False, read_folw=True):

        data_dir = os.path.join(ROOT, 'VIL100')
        self.root = data_dir
        dbfile = os.path.join(data_dir, 'data', 'db_info.yaml')
        self.imgdir = os.path.join(data_dir, 'JPEGImages')
        self.annodir = os.path.join(data_dir, 'Annotations')
        self.json = os.path.join(data_dir, 'Json')
        self.flowdir = os.path.join(data_dir, 'Flow')
        
        # extract annotation information
        with open(dbfile, 'r') as f:
            db = yaml.load(f, Loader=yaml.Loader)['sequences']
            targetset = 'train' if train else 'test'
            # targetset = 'training'
            self.info = db
            self.videos = [info['name'] for info in db if info['set'] == targetset]

        self.samples_per_video = samples_per_video if train else 1 #2
        self.sampled_frames = sampled_frames #9

        self.length = samples_per_video * len(self.videos)

        self.max_skip = max_skip #5 在一个长度为100frames的video中的采样最大间隔
        self.increment = increment

        self.transform = transform
        self.train = train
        self.calc_flow = calc_flow
        self.read_folw = read_folw
        self.max_obj = MAX_OBJECT
        self.bound = 100 #光流限制大小范围

    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, MAX_TRAINING_SKIP)

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip


    def __getitem__(self, idx):
        vid = self.videos[(idx // self.samples_per_video)] #选择在哪个video里进行采样
        # print(vid)
        imgfolder = os.path.join(self.imgdir, vid)
        annofolder = os.path.join(self.annodir, vid)
        jsonfolder = os.path.join(self.json, vid)
        flowfolder = os.path.join(self.flowdir, vid)

        frames = [name[:5] for name in os.listdir(annofolder)]
        frames.sort()
        nframes = len(frames) #100

        if self.train:
            last_sample = -1
            sample_frame = []

            nsamples = min(self.sampled_frames, nframes)
            for i in range(nsamples):
                if i == 0:
                    last_sample = random.sample(range(0, nframes - nsamples + 1), 1)[0]
                else:
                    last_sample = random.sample(
                        range(last_sample + 1, min(last_sample + self.max_skip + 1, nframes - nsamples + i + 1)), 1)[0]
                sample_frame.append(frames[last_sample])
        else:
            sample_frame = frames

        if self.transform is None:
            raise RuntimeError('Lack of proper transformation')
        
        info = {'name': vid}
        info['palette'] = Image.open(os.path.join(annofolder, frames[0]+'.png')).getpalette()
        info['size'] = np.array(Image.open(os.path.join(imgfolder, sample_frame[0] + '.jpg'))).shape[:2]
        info['ImgName'] = sample_frame
        num_obj = self.max_obj

        data = list() #training时长度为8的sample字典
        if self.calc_flow:
            maskForFlow = list()
            maskForFlow.append(cv2.imread(os.path.join(annofolder, sample_frame[0] + '.png'))[int(info['size'][0]*opt.cut_scale):, ...])
        
        for name in sample_frame: #8
            # print(name)
            sample = dict()
            sample['palette'] = info['palette'] #
            sample['img'] = np.array(Image.open(os.path.join(imgfolder, name + '.jpg')))
            sample['mask'] = np.array(Image.open(os.path.join(annofolder, name + '.png')))
            sample['img'] = sample['img'][int(info['size'][0]*opt.cut_scale):, ...] #裁减图像上面部分
            sample['mask'] = sample['mask'][int(info['size'][0]*opt.cut_scale):, ...] #裁减图像上面部分
            sample['mask'] = convert_mask(sample['mask'], self.max_obj)
            sample['flow'] = None

            if self.calc_flow:
                img = cv2.imread(os.path.join(imgfolder, name + '.jpg'))[int(info['size'][0]*opt.cut_scale):, ...]
                new_mask = cv2.imread(os.path.join(annofolder, name + '.png'))[int(info['size'][0]*opt.cut_scale):, ...]
                flow = dense_twoFrame_flow(cv2.calcOpticalFlowFarneback, img, maskForFlow[-1], new_mask, params=[0.5, 3, 15, 3, 5, 1.2, 0])
                sample['flow'] = flow
                maskForFlow.append(new_mask)
                if len(maskForFlow) > 2:
                    maskForFlow.pop(0)
            
            if self.read_folw:
                flow = np.zeros((*(info['size']), 2), dtype=np.float32)
                flow[..., 0] = cv2.imread(os.path.join(flowfolder, name + 'u.jpg'), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                flow[..., 1] = cv2.imread(os.path.join(flowfolder, name + 'v.jpg'), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                flow = flow * 2 * self.bound / 255.0 - self.bound #从灰度图恢复有精度损失[-100~100]
                flow[..., 0] = flow[..., 0] / info['size'][1]  #x-W [-1~1]
                flow[..., 1] = flow[..., 1] / info['size'][0]  #y-H [-1~1]
                flow = flow[int(info['size'][0]*opt.cut_scale):, ...]
                sample['flow'] = flow

            with open(os.path.join(jsonfolder, name+ '.jpg' + '.json')) as f:
                lanes_info = json.load(f)
                lanes_anno = []
                lanes_ids = []
                for lane in lanes_info['annotations']['lane']:
                    lanes_ids.append(lane['lane_id']-1)
                    lanes_anno.append(lane['points'])
                sample['lanes_ids'] = lanes_ids
                sample['lanes'] = lanes_anno
                cut_height_lanes(sample, int(info['size'][0] * opt.cut_scale)) #车道线点修正

            sample = self.transform(sample)
            data.append(sample)

        return data, num_obj, info

    def __len__(self):
        return self.length


def cut_height_lanes(sample, cut_height):
    if cut_height != 0:
        new_lanes = []
        for i in sample['lanes']:
            lanes = []
            for p in i:
                lanes.append((p[0], p[1] - cut_height))
            new_lanes.append(lanes)
        sample.update({'lanes': new_lanes})

DATA_CONTAINER['VIL100'] = VIL
