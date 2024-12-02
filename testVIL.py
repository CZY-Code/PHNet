import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import numpy as np
import os
import json
import time

from libs.dataset.dataV3 import ROOT, DATA_CONTAINER, multibatch_collate_fn
from libs.utils.utility import write_mask
from evaluation.generate_lane import generate_pred
from libs.dataset.transformV2 import COLORS
from evaluation.generate_lane import sample_lane

from libs.dataset.transformV3 import GenerateLaneLine
from libs.utils.lossV5 import DILaneCriterionV5
from libs.models.RouterV4 import RouterWithB
from libs.utils.config import Config
opt = Config.fromfile('./optionsV3.py')

def main():
    
    # Use CUDA
    use_gpu = torch.cuda.is_available() and int(opt.gpu_id) >= 0
    # set device
    device = torch.device('cuda:{}'.format(opt.gpu_id))
    # Data
    print('==> Preparing dataset %s' % opt.valset)
    test_transformer = GenerateLaneLine(opt.test_transforms, opt, training=False)
    testset = DATA_CONTAINER[opt.valset](
        train=False,
        transform=test_transformer,
        samples_per_video=1
    )

    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                 collate_fn=multibatch_collate_fn)
    # Model
    print("==> creating model")
    criterion = DILaneCriterionV5(cfg=opt)
    model = RouterWithB(cfg=opt, criterion=criterion)
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
         use_cuda=use_gpu,
         opt=opt,
         device=device)
    print('==> Results are saved at: {}'.format('./evaluation/txt/pred_txt'))


def test(testloader, model, use_cuda, opt, device):
    time_cost = []
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            frames, masks, lanes_lines, objs, infos, flows, occlusions = data #一个batch的数据
            frames = frames.to(device) #[1, 100, 3, 320, 640]
            masks = masks.to(device) #[1, 100, 9, 320, 640]
            lanes_lines = lanes_lines.to(device) #[1, 100, 8, 78]
            objs = objs.to(device) #tensor([8])
            flows = flows.to(device) #[1, 100, 320, 640, 2]
            occlusions = occlusions.to(device) #[1, 100, 8] #-1无 0未遮挡 1遮挡
            objs[objs == 0] = 1

            N, T, C, H, W = frames.size()
            inputs = {}
            for idx in range(N): # N=1 逐clip进行分析
                # pred = []
                inputs['frame'] = frames[idx] #[9, 3, 320, 640]
                inputs['mask'] = masks[idx] #[9, 9, 320, 640]
                inputs['lanes'] = lanes_lines[idx] #[9, 8, 78]
                inputs['lane_ids'] = inputs['lanes'][:, :, 1] #[9, 8]
                inputs['gt_flows'] = flows[idx] #FIXME
                inputs['occlusion'] = occlusions[idx] #[B, 8] 车道线阻挡与否 -1:无 0:未遮挡 1:遮挡
                inputs['num_objects'] = objs[idx]
                inputs['info'] = infos[idx]
                
                #----单次输入100帧---------
                start = time.time()
                cilp_outputs = model(inputs) #100张图的结果
                end = time.time()
                # lane_lines = cilp_outputs['lane_lines']
                # for t in range(T):
                #     generate_pred(inputs['info'], lane_lines[t], t)
                #     predlanes(infos[idx], lane_lines[t], t, show=False) #检测结果可视化
                
                #------单次输入clipLen帧--------
                # lane_lines = []
                # imgLists = [] #video太长了，进行分段预测
                # clipLen = 16 #
                # for i in range(0, T, clipLen):
                #     if i+16 <= T:
                #         imgLists.append([t for t in range(i, i+clipLen)])
                #     else:
                #         imgLists.append([t for t in range(i, T)])
                # start = time.time()
                # for i, imgList in enumerate(imgLists): # N=1 逐clip进行分析
                #     inputs['frame'] = frames[idx][imgList] #[9, 3, 320, 640]
                #     inputs['lanes'] = lanes_lines[idx][imgList] #[9, 8, 78]
                #     cilp_outputs = model(inputs) #100张图的结果
                #     lane_lines = cilp_outputs['lane_lines']
                #     for t, lanes in enumerate(lane_lines):
                #         generate_pred(inputs['info'], lanes, clipLen*i+t)
                #         # predlanes(infos[idx], lanes, clipLen*i+t) #检测结果可视化
                # end = time.time()

                tmp_time = end - start
                time_cost.append(tmp_time)
                print(inputs['info']['name']+' frames_num: ' + str(T) + ' Time cost: ' + str(tmp_time))
                print('testing fps: ' + str(1 / (tmp_time / T)))
                
                # ------可视化--------
                # generate_seg_from_line(lane_lines[0], info, t) #生成分割结果
                # predseg(outputs) #分割结果可视化
                # logits = outputs['seg']
                # out = torch.softmax(logits, dim=1)
                # pred.append(out)
                # pred = torch.cat(pred, dim=0) #[100, 9, 320, 640]
                # pred = pred.detach().cpu().numpy()
                # write_mask(pred, info, opt, directory=opt.output_dir)


def predseg(outputs: torch.Tensor, img_h, img_w):
    out_mask = torch.softmax(outputs['seg'], dim=1)
    seg_show = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    rescale_mask = F.interpolate(out_mask, (img_h, img_w))
    rescale_mask = rescale_mask[0].argmax(axis=0).detach().cpu().numpy().astype(np.uint8)
    print(rescale_mask.shape)
    print(rescale_mask.max())
    for k in range(1, rescale_mask.max()+1):
        seg_show[rescale_mask==k, :] = COLORS[k-1] #sample['palette'][(k*3):(k+1)*3]
    cv2.imshow('img_seg', seg_show)
    cv2.waitKey(0)

def predlanes(info, lanes, img_num, show=False, width=12):
    size = info['size']
    vidname = info['name']
    img_name = info['ImgName'][img_num] + '.jpg'
    json_name = img_name + '.json'
    print(vidname + img_name)
    img = cv2.imread(os.path.join(ROOT, opt.valset, 'JPEGImages',  vidname, img_name))
    seg_show = np.zeros_like(img)

    #------------生成标签---------------
    # with open(os.path.join(ROOT, opt.valset, 'JsonV4', vidname, json_name)) as f:
    #     lanes_info = json.load(f)['annotations']['lane'] #list
    #     for lane in lanes_info:
    #         points = lane['points']
    #         # points = sample_lane(points, size) #延长至图像底部
    #         if(len(points)<2): continue
    #         x_0=int(points[0][0])
    #         y_0=int(points[0][1])
    #         for i in range(1, len(points)):
    #             x_1 = int(points[i][0])
    #             y_1 = int(points[i][1])
    #             cv2.line(seg_show, (x_0, y_0), (x_1, y_1), COLORS[1], thickness=width)
    #             x_0, y_0 = x_1, y_1

    #------------生成预测----------------
    lanes_xys = []
    for lane in lanes: #pred
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x*size[1]), int(y*size[0])
            xys.append((x, y))
        lanes_xys.append(xys)
    for idx, xys in enumerate(lanes_xys):
        for i in range(1, len(xys)):
            # cv2.line(img, xys[i - 1], xys[i], COLORS[0], thickness=width)
            cv2.line(seg_show, xys[i - 1], xys[i], COLORS[1], thickness=width) #30
                
    if show:
        cv2.imshow('pred_lane', img)
        cv2.imshow('seg_show', seg_show)
        cv2.waitKey(0)
            
    if True:        
        output_name = info['ImgName'][img_num] + '.png'
        video = os.path.join('./output/VIL/', vidname)
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