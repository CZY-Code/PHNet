from matplotlib.image import imread
import numpy as np
import cv2
from argparse import ArgumentParser
import os
import yaml

def dense_twoFrame_flow(img, method, old_frame, new_frame, 
                        params=[], to_gray=True, show=False):
    # 精确方法的预处理
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    flow = method(old_frame, new_frame, None, *params) #[H, W, 2]
    # TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    # flow = TVL1.calc(old_frame, new_frame, None)

    if show:
        hsv = np.zeros((*(old_frame.shape[:2]), 3), dtype=np.uint8)
        hsv[..., 1] = 255
        # 编码:将算法的输出转换为极坐标
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # 使用色相和饱和度来编码光流
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # 转换HSV图像为BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)
        im = cv2.addWeighted(img, 1.0, bgr, 1.0, 0, dtype = -1)
        cv2.imshow('img', im)
        cv2.waitKey(0)
    
    return flow


def dense_optical_flow(method, video_path, params=[], to_gray=False):
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    # 读取第一帧
    ret, old_frame = cap.read()

    # 创建HSV并使Value为常量
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # 精确方法的预处理
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # 读取下一帧
        ret, new_frame = cap.read()
        frame_copy = new_frame
        if not ret:
            break
        # 精确方法的预处理
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        # 计算光流
        flow = method(old_frame, new_frame, None, *params) #[H,W,2]
        
        # 编码:将算法的输出转换为极坐标
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # 使用色相和饱和度来编码光流
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # 转换HSV图像为BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        old_frame = new_frame

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=["farneback", "lucaskanade_dense", "rlof"],
        required=True,
        help="Optical flow algorithm to use",
    )
    parser.add_argument(
        "--video_path", default="./input.mp4", help="Path to the video",
    )

    args = parser.parse_args()
    video_path = args.video_path
    if args.algorithm == "lucaskanade_dense":
        method = cv2.optflow.calcOpticalFlowSparseToDense
        dense_optical_flow(method, video_path, to_gray=True)
    elif args.algorithm == "farneback":
        # OpenCV Farneback算法需要一个单通道的输入图像，因此我们将BRG图像转换为灰度。
        method = cv2.calcOpticalFlowFarneback
        params = [0.5, 3, 15, 3, 5, 1.2, 0]  # Farneback的算法参数
        dense_optical_flow(method, video_path, params, to_gray=True)
    elif args.algorithm == "rlof":
    	# 与Farneback算法相比，RLOF算法需要3通道图像，所以这里没有预处理。
        method = cv2.optflow.calcOpticalFlowDenseRLOF
        dense_optical_flow(method, video_path)

def make_flow_from_VIL(store=True, bound=100):
    ROOT = '../../dataset'
    data_dir = os.path.join(ROOT, 'VIL100')
    imgdir = os.path.join(data_dir, 'JPEGImages')
    maskdir = os.path.join(data_dir, 'Annotations')
    dbfile = os.path.join(data_dir, 'data', 'db_info.yaml')
    flowdir = os.path.join(data_dir, 'Flow')
    train = True
    with open(dbfile, 'r') as f:
        db = yaml.load(f, Loader=yaml.Loader)['sequences']
        targetset = 'train' if train else 'test'
        videos = [info['name'] for info in db if info['set'] == targetset]
    
    for vid in videos: #循环每一个clip
        maskfolder = os.path.join(maskdir, vid)
        imgfolder = os.path.join(imgdir, vid)
        flowfolder = os.path.join(flowdir, vid)

        frames = [name[:5] for name in os.listdir(maskfolder)]
        frames.sort()
        if not os.path.exists(flowfolder):
            os.makedirs(flowfolder)
        
        maskForFlow = list()
        # img = cv2.imread(os.path.join(imgdir, vid, frames[0] + '.jpg'))
        # H, W, C = img.shape
        maskForFlow.append(cv2.imread(os.path.join(maskfolder, frames[0] + '.png')))

        for name in frames: #循环一帧
            print(maskfolder, name+ '.png')
            img = cv2.imread(os.path.join(imgfolder, name + '.jpg'))
            new_mask = cv2.imread(os.path.join(maskfolder, name + '.png'))
            flow = dense_twoFrame_flow(img, cv2.calcOpticalFlowFarneback, maskForFlow[-1], new_mask, params=[0.5, 3, 15, 3, 5, 1.2, 0])
            maskForFlow.append(new_mask)
            if len(maskForFlow) > 2:
                maskForFlow.pop(0)
            
            if store:
                #对两个方向的光流使用相同的截断和归一化策略
                flow = np.clip(flow, -bound, bound)
                flow = (flow + bound) * (255.0 / (2 * bound)) #在flow取之为[-bound, bound]之内进行归一化
                # flow[flow >= 255] = 255
                # flow[flow <= 0] = 0
                flow = np.round(flow).astype(np.uint8)
                
                cv2.imwrite(os.path.join(flowfolder, name+'u.jpg'), flow[:, :, 0])
                cv2.imwrite(os.path.join(flowfolder, name+'v.jpg'), flow[:, :, 1])

                # 重新载入
                # flow = np.zeros((*(img.shape[:-1]), 2), dtype=np.float32)
                # flow[..., 0] = cv2.imread(os.path.join(flowfolder, name + 'u.jpg'), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                # flow[..., 1] = cv2.imread(os.path.join(flowfolder, name + 'v.jpg'), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                # flow = flow * 2 * bound / 255.0 - bound #有精度损失


if __name__ == "__main__":
    # main()
    make_flow_from_VIL()
