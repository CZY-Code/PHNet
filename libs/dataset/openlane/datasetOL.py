#选择直接从数据集中进行数据读取
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from libs.dataset.openlane.transforms import *
from libs.dataset.openlane.utils import *
DATA_CONTAINER = {}
MAX_TRAINING_SKIP = 1 #2

def multibatch_collate_fn(batch):
    outs = [samples[0] for samples in batch]
    infos = [samples[1] for samples in batch]
    #当前batchsize = 1 batch[0]是一个长度为4的字典，存储每帧图像的信息
    frames = torch.stack([torch.stack([data['img'] for data in sample]) for sample in outs]) #[1, 4, 3, 384, 768]
    lanes_lines = torch.stack([torch.stack([torch.from_numpy(data['lane_line']) for data in sample]) for sample in outs]) #[1, 4, 4, 78]
    return frames, lanes_lines, infos

class Dataset_TrainV1(Dataset):
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

        self.datalist_video = os.listdir(self.label_root) #450
        self.datalist_video.sort()

        self.length = self.cfg.samples_per_video * len(self.datalist_video)

        # image transform
        self.transform = Transforms(cfg, self.mode == 'training')
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
    
    def increase_max_skip(self):
        self.cfg.max_skip = min(self.cfg.max_skip + self.cfg.increment, MAX_TRAINING_SKIP)
    def set_max_skip(self, max_skip):
        self.cfg.max_skip = max_skip

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
        video = self.datalist_video[(idx // self.cfg.samples_per_video)] #选择在哪个video里进行采样
        frames = [name[:-7] for name in os.listdir(os.path.join(self.label_root, video))] #[xxx.pickle]
        frames.sort()
        nframes = len(frames) #此video的总帧数100
        
        if self.mode == 'training':
            self.flip = random.randint(0, 1)
            reverse = random.randint(0, 1)
        else:
            self.flip = 0
            reverse = 0
        
        if self.mode == 'training':
            last_sample = -1
            sample_frame = []
            nsamples = min(self.cfg.sampled_frames, nframes) #min(5, 100) (16,100)
            for i in range(nsamples):
                if i == 0:
                    last_sample = random.sample(range(0, nframes - nsamples + 1), 1)[0]
                else:
                    last_sample = random.sample(
                        range(last_sample + 1, min(last_sample + self.cfg.max_skip + 1, nframes - nsamples + i + 1)), 1)[0]
                sample_frame.append(frames[last_sample])
        else:
            sample_frame = frames

        sample_frame.sort(reverse=reverse) #XXX 返回None!
        info = {'name': video}
        info['ImgName'] = sample_frame

        out = []
        for t, name in enumerate(sample_frame): #获取2+1帧数据 [0,1,2]
            img_name = os.path.join(video, name)
            img, anno = self.get_data_org(img_name)
            out.append(dict())
            out[t]['img_name'] = img_name
            out[t].update(self.get_data_aug(img, anno))
            # showData(out[t])
        
        info['size'] = (out[0]['org_h'], out[0]['org_w'])

        return out, info #([nsamples])(['img_name', 'img', 'img_rgb', 'lane_line', 'org_h', 'org_w'])

    def __len__(self):
        return self.length
DATA_CONTAINER['OpenLane'] = Dataset_TrainV1

from torchvision.transforms.functional import to_pil_image
import cv2
def showData(sample):
    img_rgb = sample['img_rgb']
    lanes = sample['lane_line']
    img_cv = np.array(to_pil_image(img_rgb))
    print(lanes)
    cv2.imshow('img', img_cv)
    cv2.waitKey(0)
    
