from libs.dataset.openlane.datasetOL import multibatch_collate_fn, DATA_CONTAINER


import torch
import cv2
import numpy as np
import os
import time
from torch.utils.data.distributed import DistributedSampler

from evaluation.generate_lane import generate_predV2
from libs.dataset.transformV2 import COLORS
from evaluation.generate_lane import sample_lane
from libs.utils.config import Config
from trainOL import seed_torch, _init_fn

# from libs.utils.loss4OL import Criterion4OL
# from libs.utils.loss4OLV2 import Criterion4OL
from libs.utils.loss4OLV3 import Criterion4OL

from libs.models.Router4OL import RouterOL
# from libs.models.Router4OLV3 import RouterOL
opt = Config.fromfile('./options4OL.py')
# opt = Config.fromfile('./options4OLV2.py')
ROOT = opt.root

def main():
    seed_torch()
    # Use CUDA
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Data
    print('==> Preparing dataset %s' % opt.valset)
    testset = DATA_CONTAINER[opt.valset](cfg=opt.dscfg, mode='validation')

    # testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt.workers,
    #                              collate_fn=multibatch_collate_fn)
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=1, #1 目前在这里只能最多取16张图像
                                            sampler=DistributedSampler(testset, shuffle=False),
                                            pin_memory=True,
                                            num_workers=8,
                                            collate_fn=multibatch_collate_fn,
                                            drop_last=False,
                                            worker_init_fn=_init_fn)
    # Model
    print("==> creating model")
    criterion = Criterion4OL(cfg=opt)
    model = RouterOL(cfg=opt, criterion=criterion)
    model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    with torch.cuda.device(device):
        model = model.to(device)
    # set training parameters
    for p in model.parameters():
        p.requires_grad = False

    if opt.initial_model:
        print('==> Resuming from checkpoint {}'.format(opt.initial_model))
        assert os.path.isfile(opt.initial_model), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.initial_model, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    elif opt.resume_model:
        print('==> Resuming from checkpoint {}'.format(opt.resume_model))
        assert os.path.isfile(opt.resume_model), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.resume_model, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    print('==> Runing model on dataset {}, totally {:d} videos'.format(opt.valset, len(testloader)))
    test(testloader,
         model=model,
         opt=opt,
         device=device)
    print('==> Results are saved at: {}'.format('./evaluation/txt4OL/pred_txt'))


def test(testloader, model, opt, device):
    time_cost = []
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            frames, lanes_lines, infos = data #一个batch的数据
            frames = frames.to(device) #[1, 100, 3, 320, 640]
            lanes_lines = lanes_lines.to(device) #[1, 100, 8, 78]
            N, T, C, H, W = frames.size()
            inputs = {}
            for idx in range(N):
                info = infos[idx]
                lane_lines = []
                imgLists = [] #video太长了，进行分段预测

                # TList = list(range(T))
                # clipLen = T//2
                # if T > 100:
                #     imgLists.append(TList[:T//2])
                #     imgLists.append(TList[T//2:])
                # else:
                #     imgLists.append(TList)

                clipLen = 16
                for i in range(0, T, clipLen):
                    if i+16 <= T:
                        imgLists.append([t for t in range(i, i + clipLen)])
                    else:
                        imgLists.append([t for t in range(i, T)])

                start = time.time()
                for i, imgList in enumerate(imgLists): # N=1 逐clip进行分析
                    inputs['frame'] = frames[idx][imgList] #[9, 3, 320, 640]
                    inputs['lanes'] = lanes_lines[idx][imgList] #[9, 8, 78]
                    # inputs['lane_ids'] = inputs['lanes'][:, :, 1] #[9, 8]
                    cilp_outputs = model(inputs) #100张图的结果
                    lane_lines = cilp_outputs['lane_lines']
                    for t, lanes in enumerate(lane_lines):
                        # generate_predV2(info, lanes, clipLen*i+t)
                        predlanesV2(info, lanes, clipLen*i+t) #检测结果可视化
                end = time.time()                
                tmp_time = end - start
                time_cost.append(tmp_time)
                print(info['name']+' frames_num: ' + str(T) + ' Time cost: ' + str(tmp_time))
                print('testing fps: ' + str(1 / (tmp_time / T)))
                
                # ------可视化--------
                # generate_seg_from_line(lane_lines[0], info, t) #生成分割结果
                # predlanes(lane_lines[0], info, t) #检测结果可视化
                # write_mask(pred, info, opt, directory=opt.output_dir)


def predlanesV2(info, lanes, img_num, show=False, width=12):
    size = info['size']
    vidname = info['name']
    img_name = info['ImgName'][img_num] + '.jpg'
    label_name = info['ImgName'][img_num] + '.lines.txt'
    # print(vidname + img_name)

    labelTxtName = os.path.join('./evaluation/txt4OL/anno_txt', vidname, label_name)
    img = cv2.imread(os.path.join('./dataset/OpenLane/images/validation', vidname, img_name))
    seg_show = np.zeros_like(img)

    #---------------标签-------------
    # with open(labelTxtName, 'r') as f:
    #     lanes_info = f.readlines()
    #     for lane in lanes_info: #循环单张图的lane
    #         laneStr = lane.split()
    #         for i in range(3, len(laneStr), 2):
    #             # cv2.line(img, (int(float(laneStr[i-3])), int(float(laneStr[i-2]))), 
    #             #          (int(float(laneStr[i-1])), int(float(laneStr[i]))), COLORS[1], thickness=width)
    #             cv2.line(seg_show, (int(float(laneStr[i-3])*2), int(float(laneStr[i-2])*2)), 
    #                      (int(float(laneStr[i-1])*2), int(float(laneStr[i])*2)), COLORS[1], thickness=width)
                
        # for lane in lanes_info:
        #     points = lane['points']
        #     # points = sample_lane(points, size) #延长至图像底部
        #     if(len(points)<2): continue
        #     x_0=int(points[0][0])
        #     y_0=int(points[0][1])
        #     for i in range(1, len(points)):
        #         x_1 = int(points[i][0])
        #         y_1 = int(points[i][1])
        #         cv2.line(img, (x_0, y_0), (x_1, y_1), COLORS[1], thickness=width)
        #         x_0, y_0 = x_1, y_1
    
    lanes_xys = []
    for lane in lanes: #pred
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x*size[1]), int(y*size[0]+480)
            xys.append((x, y))
        lanes_xys.append(xys)
    for idx, xys in enumerate(lanes_xys):
        for i in range(1, len(xys)):
            # cv2.line(img, xys[i - 1], xys[i], COLORS[0], thickness=width)
            cv2.line(seg_show, xys[i - 1], xys[i], COLORS[1], thickness=width)
                
    if show:
        cv2.imshow('pred_lane', img)
        cv2.imshow('seg_show', seg_show)
        cv2.waitKey(0)
            
    if True:        
        output_name = info['ImgName'][img_num] + '.png'
        video = os.path.join('./output/openlane/', vidname)
        if not os.path.exists(video):
            os.makedirs(video)
        cv2.imwrite(os.path.join(video, output_name), seg_show)

def generate_seg_from_line(lanes, info, img_num, show=False, width=4, directory=opt.output_dir):
    size = info['size']
    vidname = info['name']
    img_name = info['ImgName'][img_num] + '.jpg'
    print(vidname + img_name)
    img = cv2.imread(os.path.join(ROOT, opt.valset, 'JPEGImages', vidname, img_name))
    seg_show = np.zeros_like(img)

    lanes_xys = []
    for lane in lanes: #pred
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x*size[1]), int(y*size[0])
            xys.append((x, y))
        lanes_xys.append(xys)
    lanes_xys.sort(key=lambda xys : xys[-1][0]) #靠近下边缘的点的横坐标排序
    for idx, xys in enumerate(lanes_xys):
        for i in range(1, len(xys)):
            cv2.line(img, xys[i - 1], xys[i], COLORS[0], thickness=width)
            cv2.line(seg_show, xys[i - 1], xys[i], COLORS[idx], thickness=20) #30
    
    if show:
        cv2.imshow('pred', img)
        cv2.imshow('seg',seg_show)
        cv2.waitKey(0)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    video = os.path.join(directory, vidname)
    if not os.path.exists(video):
        os.makedirs(video)

    output_name = info['ImgName'][img_num] + '.png'
    video = os.path.join(directory, vidname)
    cv2.imwrite(os.path.join(video, output_name), seg_show)
    

if __name__ == '__main__':
    main()
