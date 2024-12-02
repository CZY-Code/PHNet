import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from libs.dataset.openlane.transforms import *
from libs.dataset.openlane.utils import *
DATA_CONTAINER = {}
# MAX_TRAINING_SKIP = 5 #2

def multibatch_collate_fn(batch):
    outs = [samples[0] for samples in batch]
    infos = [samples[1] for samples in batch]
    #当前batchsize = 1 batch[0]是一个长度为4的字典，存储每帧图像的信息
    frames = torch.stack([torch.stack([data['img'] for data in sample]) for sample in outs]) #[1, 4, 3, 384, 768]
    lanes_lines = torch.stack([torch.stack([torch.from_numpy(data['lane_line']) for data in sample]) for sample in outs]) #[1, 4, 4, 78]
    return frames, lanes_lines, infos

class Dataset_TrainV2(Dataset):
    def __init__(self, cfg, mode=None):
        assert mode is not None
        self.mode = mode
        self.cfg = cfg
        if self.mode == 'training':
            self.img_root = './dataset/OpenLane/images/training'
            self.label_root = './dataset/OpenLane/OpenLane-V/label/training'
        else:
            self.img_root = './dataset/OpenLane/images/validation'
            self.label_root = './dataset/OpenLane/OpenLane-V/label/validation'
            self.cfg.samples_per_video = 1

        self.datalist_video = load_pickle(f'/home/chengzy/MMA-TR-Net/dataset/OpenLane/OpenLane-V/list/datalist_training_{self.cfg.clip_length}')
        self.datalist = list(self.datalist_video) #长度为59174
        self.length = len(self.datalist) // self.cfg.split_dataset

        # image transform
        self.transform = Transforms(cfg, self.mode == 'training')
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def cropping(self, img, lanes):
        img = img.crop((0, self.cfg.crop_size, int(img.size[0]), int(img.size[1]))) #裁掉图像最上方crop_size高度的天空
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
        img = Image.open(f'{self.cfg.dir["dataset"]}/images/{self.mode}/{img_name}.jpg').convert('RGB')
        # dict_keys(['lanes']) 长度为车道线数的[points, 2] [x,y] 图像原点在左上角
        anno = load_pickle(f'{self.cfg.dir["dataset"]}/OpenLane-V/label/{self.mode}/{img_name}') 
        img, anno = self.cropping(img, anno)
        return img, anno

    def get_data_aug(self, img, anno):
        # if self.mode == 'training':
        #     img_new, anno_new = self.transform.process(img, anno['lanes'])
        # else:
        #     img_new, anno_new = self.transform.process_for_test(img, anno['lanes'])
        img_new, anno_new = self.transform.process(img, anno['lanes'])
        img_new = Image.fromarray(img_new)
        img_new = self.to_tensor(img_new)
        self.org_width, self.org_height = img.size
        return {'img': self.normalize(img_new),
                'img_rgb': img_new,
                'lane_line': anno_new['label'],
                'org_h': self.org_height, 'org_w': self.org_width}

    def __getitem__(self, idx):
        index = idx * self.cfg.split_dataset + random.randint(0, self.cfg.split_dataset - 1)
        t_frame = self.datalist[index]
        self.flip = random.randint(0, 1)
        reverse = random.randint(0, 1)
        if reverse == 0: #包含self.cfg.clip_length + 1
            datalist_video = sorted(random.sample(self.datalist_video[t_frame], self.cfg.clip_length + 1), reverse=True)
        else:
            datalist_video = sorted(random.sample(self.datalist_video[t_frame], self.cfg.clip_length + 1), reverse=False)
        info = {'name': t_frame}
        info['ImgName'] = datalist_video
        out = []
        for t in range(self.cfg.clip_length + 1): #获取2+1帧数据 [0,1,2]
            img_name = datalist_video[t]
            img, anno = self.get_data_org(img_name)
            out.append(dict())
            out[t]['img_name'] = img_name
            out[t].update(self.get_data_aug(img, anno))
            # showData(out[t])
        info['size'] = (out[0]['org_h'], out[0]['org_w'])
        return out, info #([nsamples])(['img_name', 'img', 'img_rgb', 'lane_line', 'org_h', 'org_w'])

    def __len__(self):
        return self.length
    
DATA_CONTAINER['OpenLane'] = Dataset_TrainV2

def showData(sample):
    from torchvision.transforms.functional import to_pil_image
    import cv2
    img_rgb = sample['img_rgb']
    lanes = sample['lane_line']
    img_cv = np.array(to_pil_image(img_rgb))
    print(lanes)
    cv2.imshow('img', img_cv)
    cv2.waitKey(0)
    
