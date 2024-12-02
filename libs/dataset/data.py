import torch
import os
import numpy as np
import torch
import json
import yaml
import random

from PIL import Image
from torch.utils.data import Dataset
from optionsV2 import OPTION as opt

DATA_CONTAINER = {}
ROOT = opt.root
MAX_OBJECT = opt.max_object #车道线分割实例数目
MAX_TRAINING_SKIP = 100


def multibatch_collate_fn(batch):
    min_time = min([sample[0].shape[0] for sample in batch])
    frames = torch.stack([sample[0] for sample in batch])
    masks = torch.stack([sample[1] for sample in batch])

    objs = [torch.LongTensor([sample[2]]) for sample in batch]
    objs = torch.cat(objs, dim=0)

    try:
        info = [sample[3] for sample in batch]
    except IndexError as ie:
        info = None
    # print(frames.shape)
    # print(masks.shape)
    # print(objs)
    # print(len(info))
    # exit(0)
    return frames, masks, objs, info

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
                 transform=None, max_skip=5, increment=5, samples_per_video=10):

        data_dir = os.path.join(ROOT, 'VIL100')

        dbfile = os.path.join(data_dir, 'data', 'db_info.yaml')
        self.imgdir = os.path.join(data_dir, 'JPEGImages')
        self.annodir = os.path.join(data_dir, 'Annotations')
        self.json = os.path.join(data_dir, 'Json')

        self.root = data_dir

        # extract annotation information
        with open(dbfile, 'r') as f:
            db = yaml.load(f, Loader=yaml.Loader)['sequences']
            targetset = 'train' if train else 'test'
            # targetset = 'training'
            self.info = db
            self.videos = [info['name'] for info in db if info['set'] == targetset]

        self.samples_per_video = samples_per_video #2
        self.sampled_frames = sampled_frames #9

        self.length = samples_per_video * len(self.videos)

        self.max_skip = max_skip #5 在一个长度为100frames的video中的采样最大间隔
        self.increment = increment

        self.transform = transform
        self.train = train
        self.max_obj = MAX_OBJECT

    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, MAX_TRAINING_SKIP)

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip


    def __getitem__(self, idx):
        vid = self.videos[(idx // self.samples_per_video)] #选择在哪个video里进行采样

        imgfolder = os.path.join(self.imgdir, vid)
        annofolder = os.path.join(self.annodir, vid)
        jsonfolder = os.path.join(self.json, vid)

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
                        range(last_sample + 1, min(last_sample + self.max_skip + 1, nframes - nsamples + i + 1)),
                        1)[0]
                sample_frame.append(frames[last_sample])
        else:
            sample_frame = frames

        frame = [np.array(Image.open(os.path.join(imgfolder, name + '.jpg'))) for name in sample_frame]
        mask = [np.array(Image.open(os.path.join(annofolder, name + '.png'))) for name in sample_frame]
        anno_lanes = []
        for name in sample_frame:
            with open(os.path.join(jsonfolder, name+ '.jpg' + '.json')) as f:
                lanes_info = json.load(f)
                # lanes = lanes_info['annotations']['lane']
                lanes_anno = []
                for lane in lanes_info['annotations']['lane']:
                    lanes_anno.append(lane['points'])
                anno_lanes.append(lanes_anno)

        mask = [convert_mask(msk, self.max_obj) for msk in mask]

        info = {'name': vid}
        info['palette'] = Image.open(os.path.join(annofolder, frames[0]+'.png')).getpalette()
        info['size'] = frame[0].shape[:2]
        info['ImgName'] = frames

        if self.transform is None:
            raise RuntimeError('Lack of proper transformation')

        frame, mask = self.transform(frame, mask, False)

        num_obj = self.max_obj

        return frame, mask, num_obj, info

    def __len__(self):
        return self.length


DATA_CONTAINER['VIL100'] = VIL
