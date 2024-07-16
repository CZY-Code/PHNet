
# img_h = 320
# img_w = 640
img_h = 384
img_w = 768

trainset = ['VIL100']
valset = 'VIL100'
setting = '60_lr0.001deay1e-6_sgd'
root = './dataset'  # dataset root path
datafreq = [3] #
max_object = 8 # max number of instances
input_size = (img_h, img_w)   # input image size
sampled_frames = 16 #16   # min sampled time length while trianing
max_skip = [5]         # max skip time length while trianing
samples_per_video = 2  # sample numbers per video

# ----------------------------------------- model configuration ---------------------------------------------
keydim = 128
valdim = 512
save_freq = 1
save_freq_max = 5
epochs_per_increment = 2

# ---------------------------------------- training configuration -------------------------------------------
epochs = 50 #60
train_batch = 1
learning_rate = 5e-4 #1e-3 记得改！
gamma = 0.1
momentum = (0.9, 0.999)
solver = 'adamW'             # 'sgd' or 'adam' 'adamW'
weight_decay = 1e-3  #12e-4
layer_decay = 0.9
milestone = []              # epochs to degrades the learning rate

# ---------------------------------------- testing configuration --------------------------------------------
epoch_per_test = 5

# ------------------------------------------- other configuration -------------------------------------------
checkpoint = 'models'
initial = ''      # path to initialize the backbone
initial_model = './models/VIL100/60_lr0.001deay1e-6_sgd/model/50.pth.tar'
resume_model = ''#'./models/VIL100/60_lr0.001deay1e-6_sgd/model/model_best.pth.tar'
gpu_id = '0'      # defualt gpu-id (if not specified in cmd)
workers = 0       #1
save_indexed_format = True # set True to save indexed format png file, otherwise segmentation with original image
output_dir = './output'

#-----------------------------
transforms = [dict(name='Resize',
                   parameters=dict(size=dict(height=img_h, width=img_w)), 
                   p=1.0),
              dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.1), #p=0.1
              dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1), #p=0.1
              dict(name='MultiplyAndAddToBrightness',
                   parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                   p=0.5), #p=0.6
              dict(name='AddToHueAndSaturation',
                   parameters=dict(value=(-10, 10)), p=0.5), #p=0.7
          #     dict(name='OneOf',
          #          transforms = [
          #             dict(name='EdgeDetect', parameters=dict(alpha=(0, 0.7))),
          #             dict(name='DirectedEdgeDetect', parameters=dict(alpha=(0, 0.7), direction=(0.0, 1.0)))
          #          ],
          #          p=0.2),
          #     dict(name='OneOf',
          #          transforms=[
          #              dict(name='Dropout', parameters=dict(p=(0.01, 0.1), per_channel=0.5)),
          #              dict(name='CoarseDropout', parameters=dict(p=(0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2))
          #          ],
          #          p=0.2),
              dict(name='OneOf',
                   transforms=[dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                               dict(name='MedianBlur', parameters=dict(k=(3, 5))),
                              #  dict(name='GaussianBlur', parameters=dict(sigma=(0, 3.0)))
                               ],
                   p=0.2), #p=0.2
              dict(name='Affine',
                   parameters=dict(translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)),
                                   rotate=(-3, 3), scale=(0.95, 1.05)), p=0.5), #p=0.5
              dict(name='Resize', 
                   parameters=dict(size=dict(height=img_h, width=img_w)),
                   p=1.0),]

test_transforms = [dict(name='Resize',
                        parameters=dict(size=dict(height=img_h, width=img_w)),
                        p=1.0),]

num_points = 36 #72
n_offsets = 36 #72
max_lanes = 8
cut_scale = 0.35

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

cls_weight = 2.5 #8.0
reg_weight = 0.5 #0.5
iou_weight = 2.0 #1.5

#nms_thres=50
test_parameters = dict(conf_threshold=0.6, nms_thres=50, nms_topk=max_lanes)

num_classes = max_lanes + 1
ignore_label = 255
bg_weight = 0.4