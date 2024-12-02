import sys
import os
import numpy as np
from skimage.io import imread, imshow
from collections import deque
from video_metrics import *

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def __len__(self):
        return len(self.deq)

    def reset(self):
        self.deq = deque(maxlen=480)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        for i in range(n):
            self.deq.append(val)
        self.val = val
        self.sum = np.sum(self.deq)
        self.count = len(self.deq)
        self.avg = self.sum / self.count

def measure_video():
    output_dir = '../output'
    label_dir = '../dataset/VIL100/Annotations'
    maskfolders = os.listdir(output_dir)
    maskfolders.sort()
    J_list = AverageMeter()
    F_list = AverageMeter()
    for folder in maskfolders:
        masklist = os.listdir(os.path.join(output_dir, folder))
        masklist.sort()
        j_measure = AverageMeter()
        f_measure = AverageMeter()
        for maskname in masklist:
            maskpath = os.path.join(output_dir, folder, maskname)
            labelpath = os.path.join(label_dir, folder, maskname)
            pred = imread(maskpath, as_gray=True) #(H, W)
            label = imread(labelpath, as_gray=True)

            mask = np.where(pred>0.0, 1, 0).astype(np.bool) #将车道线看作一个类别?
            gt = np.where(label>0.0, 1, 0).astype(np.bool)
            f_measure.update(db_eval_boundary(mask, gt))

            bimask = np.stack((~mask, mask), axis=-1)
            bigt = np.stack((~gt, gt), axis=-1)
            j_measure.update(db_eval_iou(bimask, bigt))
            
        print(j_measure.avg)
        print(f_measure.avg)
        J_list.update(j_measure.avg)
        F_list.update(f_measure.avg)
    print(J_list.avg)
    print(F_list.avg)

if __name__ == '__main__':
    measure_video()