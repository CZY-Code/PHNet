import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from libs.dataset.openlane.transforms import *
from libs.dataset.openlane.utils import *

def multibatch_collate_fn(batch): 
    #batch = 1 batch[0]是一个长度为4的字典，存储每帧图像的信息
    frames = torch.stack([torch.stack([data['img'] for data in sample]) for sample in batch]) #[1, 4, 3, 384, 768]
    lanes_lines = torch.stack([torch.stack([torch.from_numpy(data['lane_line']) for data in sample]) for sample in batch]) #[1, 4, 4, 78]
    return frames, lanes_lines

class Dataset_Train(Dataset):
    def __init__(self, cfg, update=None):
        self.cfg = cfg
        # datalist_split_video长度为 622  list datalist_{3}长度为66266
        self.datalist_video = load_pickle(f'{self.cfg.dir["pre3_train"]}/datalist_{3}') #长度为66266 datalist_video['xxx'] = 长度为4的list
        self.datalist = list(self.datalist_video) #datalist_video的keys()

        if update == True:
            err = load_pickle(f'{self.cfg.dir["model1"]}/val_for_training_set/pickle/error')
            datalist = load_pickle(f'{self.cfg.dir["model1"]}/val_for_training_set/pickle/datalist')
            idx_sorted = np.argsort(err)[::-1]
            idx_sorted = idx_sorted[:int(len(idx_sorted) * 0.3)]
            errorlist = list(np.array(datalist)[idx_sorted])
            errorlist = sorted(list(set(errorlist).intersection(set(self.datalist))))
            ratio = int(np.round(len(self.datalist) / (len(errorlist))))
            print(f'val for training set ratio : {ratio}, total num : {len(self.datalist)}, error num : {len(errorlist)}')
            datalist = self.datalist + (errorlist * ratio)
            np.random.shuffle(datalist)
            self.datalist = datalist[:len(self.datalist)]

        # image transform
        self.transform = Transforms(cfg)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def cropping(self, img, lanes):
        img = img.crop((0, self.cfg.crop_size, int(img.size[0]), int(img.size[1]))) #裁掉图像最上方ccrop_size高度的天空
        for i in range(len(lanes['lanes'])):
            if len(lanes['lanes'][i]) == 0: #车道线长度为0则跳过
                continue
            lanes['lanes'][i][:, 1] -= self.cfg.crop_size #所有车道线点的y值-crop_size，因为原点为左上角
            if self.flip == 1:
                lanes['lanes'][i][:, 0] = (self.cfg.org_width - 1) - lanes['lanes'][i][:, 0]
        if self.flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        return img, lanes

    def get_data_org(self, img_name):
        img = Image.open(f'{self.cfg.dir["dataset"]}/images/training/{img_name}.jpg').convert('RGB')
        #dict_keys(['lanes']) 长度为车道线数的[points, 2] [x,y] 图像原点在左上角
        anno = load_pickle(f'{self.cfg.dir["dataset"]}/OpenLane-V/label/training/{img_name}') 
        img, anno = self.cropping(img, anno)
        return img, anno

    def get_data_aug(self, img, anno):
        img_new, anno_new = self.transform.process(img, anno['lanes'])
        img_new = Image.fromarray(img_new)
        img_new = self.to_tensor(img_new)
        self.org_width, self.org_height = img.size
        return {'img': self.normalize(img_new),
                'img_rgb': img_new,
                'lane_line': anno_new['label'],
                'org_h': self.org_height, 'org_w': self.org_width}


    def __getitem__(self, idx):
        out = []
        t_frame = self.datalist[idx]
        self.flip = random.randint(0, 1)
        reverse = random.randint(0, 1)
        if reverse == 0: #包含self.cfg.clip_length + 1
            datalist_video = sorted(random.sample(self.datalist_video[t_frame], self.cfg.clip_length + 1), reverse=True)
        else:
            datalist_video = sorted(random.sample(self.datalist_video[t_frame], self.cfg.clip_length + 1), reverse=False)

        for t in range(self.cfg.clip_length + 1): #获取2+1帧数据 [0,1,2]
            img_name = datalist_video[t]
            img, anno = self.get_data_org(img_name)
            out.append(dict())
            out[t]['img_name'] = img_name
            out[t].update(self.get_data_aug(img, anno))

        return out #(['t-0', 't-1', 't-2']) (['img_name', 'img', 'img_rgb', 'lane_line', 'org_h', 'org_w'])

    def __len__(self):
        return len(self.datalist)

class Dataset_Test(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datalist = load_pickle(f'{self.cfg.dir["dataset"]}/OpenLane-V/list/datalist_validation')
        self.datalist_video = load_pickle(f'{self.cfg.dir["pre3_test"]}/datalist_{3}')

        if cfg.sampling == True:
            if cfg.sampling_mode == 'video':
                datalist_out = list()
                datalist_video_out = dict()
                self.datalist_video = load_pickle(f'{self.cfg.dir["pre3_test"]}/datalist_{3}')
                self.datalist_split_video = load_pickle(f'{self.cfg.dir["pre3_test"]}/datalist_split_video')
                sampling = np.arange(0, len(self.datalist_split_video), cfg.sampling_step)
                datalist_video = np.array(list(self.datalist_split_video))[sampling].tolist()
                for i in range(len(datalist_video)):
                    video_name = datalist_video[i]
                    datalist_out += self.datalist_split_video[video_name]
                for i in range(len(datalist_out)):
                    datalist_video_out[datalist_out[i]] = self.datalist_video[datalist_out[i]]
                self.datalist_video = datalist_video_out
                self.datalist = list(self.datalist_video)

        # image transform
        self.transform = Transforms(cfg)
        self.transform.settings()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def cropping(self, img, lanes):
        img = img.crop((0, self.cfg.crop_size, int(img.size[0]), int(img.size[1])))
        for i in range(len(lanes['lanes'])):
            if len(lanes['lanes'][i]) == 0:
                continue
            lanes['lanes'][i][:, 1] -= self.cfg.crop_size
        return img, lanes

    def get_data_org(self, idx):
        img = Image.open(f'{self.cfg.dir["dataset"]}/images/validation/{self.datalist[idx]}.jpg').convert('RGB')
        anno = load_pickle(f'{self.cfg.dir["dataset"]}/OpenLane-V/label/validation/{self.datalist[idx]}')
        img, anno = self.cropping(img, anno)
        return img, anno

    def get_data_aug(self, img, anno):
        img_new, anno_new = self.transform.process_for_test(img, anno['lanes'])

        img_new = Image.fromarray(img_new)
        img_new = self.to_tensor(img_new)
        self.org_width, self.org_height = img.size

        return {'img': self.normalize(img_new),
                'img_rgb': img_new,
                'lanes': anno_new,
                'org_h': self.org_height, 'org_w': self.org_width}

    def get_downsampled_label_seg(self, lanes, idx, sf):
        for s in sf:
            lane_pts = np.copy(lanes)
            lane_pts[:, 0] = lanes[:, 0] / (self.cfg.width - 1) * (self.cfg.width // s - 1)
            lane_pts[:, 1] = lanes[:, 1] / (self.cfg.height - 1) * (self.cfg.height // s - 1)

            self.label['seg_label'][s] = cv2.polylines(self.label['seg_label'][s], [np.int32(lane_pts)], False, 1, self.cfg.lane_width['seg'])

    def get_label(self, data):
        out = dict()

        self.label = dict()
        self.label['org_label'] = np.ascontiguousarray(np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8))
        self.label['seg_label'] = dict()

        for s in self.cfg.scale_factor['seg']:
            self.label['seg_label'][s] = np.ascontiguousarray(np.zeros((self.cfg.height // s, self.cfg.width // s), dtype=np.float32))

        for i in range(len(data['lanes'])):
            lane_pts = data['lanes'][i]
            self.label['org_label'] = cv2.polylines(self.label['org_label'], [np.int32(lane_pts)], False, 1, self.cfg.lane_width['org'], lineType=cv2.LINE_AA)
            self.get_downsampled_label_seg(lane_pts, i, self.cfg.scale_factor['seg'])

        for s in self.cfg.scale_factor['seg']:
            if self.cfg.lane_width['mode'] == 'gaussian':
                self.label['seg_label'][s] = cv2.GaussianBlur(self.label['seg_label'][s], self.cfg.lane_width['kernel'],
                                                              sigmaX=self.cfg.lane_width['sigmaX'], sigmaY=self.cfg.lane_width['sigmaY'])
            else:
                self.label['seg_label'][s] = cv2.dilate(self.label['seg_label'][s], kernel=self.cfg.lane_width['kernel'], iterations=1)

        for s in self.cfg.scale_factor['seg']:
            self.label['seg_label'][s] = np.int64(self.label['seg_label'][s] != 0)

        self.label['org_label'] = np.float32(self.label['org_label'] != 0)

        out.update(self.label)

        return out

    def remove_dict_keys(self, data):
        data.pop('lanes')
        return data

    def __getitem__(self, idx):
        out = dict()
        out['img_name'] = self.datalist[idx]
        out['prev_num'] = len(self.datalist_video[self.datalist[idx]]) - 1
        img, anno = self.get_data_org(idx)
        out.update(self.get_data_aug(img, anno))
        out.update(self.get_label(out))
        out = self.remove_dict_keys(out)

        return out

    def __len__(self):
        return len(self.datalist)
