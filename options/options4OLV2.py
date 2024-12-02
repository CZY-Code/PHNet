import os
import numpy as np

# img_h = 320
# img_w = 640
img_h = 384
img_w = 768

trainset = ['OpenLane']
valset = 'OpenLane'
setting = '60_lr0.001deay1e-4_adamW'
root = './dataset'  # dataset root path
datafreq = [1] #[4]
input_size = (img_h, img_w)   # input image size

# ----------------------------------------- model configuration ---------------------------------------------
save_freq_max = 5
#每epochs_per_increment个epoch max_skip增加increment
# epochs_per_increment = 15

# ---------------------------------------- training configuration -------------------------------------------
epochs = 20 #60
train_batch = 1
learning_rate = 1e-4 #1e-3 记得改！
gamma = 0.1
momentum = (0.9, 0.999)
solver = 'adamW'             # 'sgd' or 'adam' 'adamW'
weight_decay = 1e-4  #5e-4
layer_decay = 0.9
milestone = []              # epochs to degrades the learning rate

# ---------------------------------------- testing configuration --------------------------------------------
epoch_per_test = 10 #5

# ------------------------------------------- other configuration -------------------------------------------
checkpoint = 'models'
initial = ''      # path to initialize the backbone
initial_model = './models/OpenLane/60_lr0.001deay1e-4_adamW/model/50.pth.tar'
resume_model = ''#'./models/OpenLane/60_lr0.001deay1e-4_adamW/model/model_best.pth.tar'
gpu_id = '0'      # defualt gpu-id (if not specified in cmd)
workers = 0       #1
save_indexed_format = True # set True to save indexed format png file, otherwise segmentation with original image
output_dir = './output'

num_priors = 240
num_points = 72
n_offsets = 72
max_lanes = 4
# cut_scale = 0.375 #480/1280

backbone = 'resnet' #
backbone = dict(resnet='resnet18',
                pretrained=True,
                replace_stride_with_dilation=[False, False, False],
                out_conv=False,
                # norm_layer='FrozenBatchNorm2d'
               )

neck = dict(in_channels=[128, 256, 512],
            out_channels=64,
            num_outs=3,
            attention=False)

cls_weight = 8.0 #8.0
reg_weight = 0.5 #0.5
iou_weight = 1.5 #1.5

#conf_threshold=0.4 nms_thres=50
test_parameters = dict(conf_threshold=0.35, nms_thres=50, nms_topk=max_lanes)

class DSconfig(object):
    def __init__(self):
        # --------basics-------- #
        self.setting_for_path()
        self.setting_for_image_param()
        self.setting_for_dataloader()
        # --------modeling-------- #
        self.setting_for_evaluation()

        self.setting_for_video_processing()
        #-------for detection---------
        self.setting_for_detection()

    def setting_for_path(self):
        self.pc = 'main'
        self.dir = dict()

        self.dir['dataset'] = '--dataset path'
        self.dir['proj'] = os.path.dirname(os.getcwd()) + '/'
        # ------------------- need to modify ------------------- #
        self.dir['head_pre'] = '--preprocessed data path'
        # ------------------------------------------------------ #
        self.dir['pre2'] = f'{self.dir["head_pre"]}/P02_SVD/output_training/pickle'
        self.dir['pre3_train'] = f'{self.dir["head_pre"]}/P03_video_based_datalist/output_training/pickle'
        self.dir['pre3_test'] = f'{self.dir["head_pre"]}/P03_video_based_datalist/output_validation/pickle'

        self.dir['out'] = f'{os.getcwd().replace("code", "output")}'
        self.dir['weight'] = f'{self.dir["out"]}/train/weight'
        self.dir['weight_paper'] = '--pretrained data path'

    def setting_for_image_param(self):
        self.org_height = 1280
        self.org_width = 1920
        self.height = img_h
        self.width = img_w
        self.size = [self.width, self.height, self.width, self.height]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.crop_size = 480

    def setting_for_dataloader(self):
        self.data_flip = True
        self.mode_transform = 'complex'  # ['custom', 'basic', 'complex']
        self.sampling = False
        self.sampling_step = 5
        self.sampling_mode = 'video'  # ['video', 'image']
        self.update_datalist = True

    def setting_for_video_processing(self):
        self.num_t = 1  # use previous {} frames
        self.window_size = 5
        self.clip_length = 15
        self.split_dataset = 50

    def setting_for_evaluation(self):
        self.eval_h = self.org_height // 2
        self.eval_w = self.org_width // 2

    def setting_for_detection(self):
        self.max_lane_num = max_lanes
        self.run_mode = 'train'
        self.dir['dataset'] = './dataset/OpenLane'
        pre_dir = './dataset/OpenLane/preprocessed/OpenLane-V'
        if pre_dir is not None:
            self.dir['head_pre'] = pre_dir
            self.dir['pre2'] = self.dir['pre2'].replace('--preprocessed data path', pre_dir)
            self.dir['pre3_train'] = self.dir['pre3_train'].replace('--preprocessed data path', pre_dir)
            self.dir['pre3_test'] = self.dir['pre3_test'].replace('--preprocessed data path', pre_dir)
        self.num_points = num_points
        self.n_offsets = self.num_points
        self.n_strips = self.num_points - 1
        self.strip_size = self.height / self.n_strips
        self.offsets_ys = np.arange(self.height, -1, -self.strip_size)
        # self.max_skip = 1 #5 # max skip time length while trianing
        # self.sampled_frames = 16 # min sampled time length while trianing
        # self.samples_per_video = 1 #2  # sample numbers per video
        # self.increment=1 #5 

dscfg = DSconfig()