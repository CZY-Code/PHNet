# Parallel Heterogeneous Networks with Adaptive Routing for Online Video Lane Detection

### Requirements
- PyTorch >= 1.10
- CUDA >= 10.0
- CuDNN >= 7.6.5
- python >= 3.6

### Installation
1. Download repository. We call this directory as `ROOT`:
```
$ git clone https://github.com/CZY-Code/PHNet.git
```
2. Install dependencies:
```
$ conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
$ pip install -r requirements.txt
```
Pytorch can be installed on [here](https://pytorch.org/get-started/previous-versions/). Other versions might be available as well.

### Dataset
Download [OpenLane-V](https://github.com/dongkwonjin/RVLD) and [VIL-100](https://github.com/yujun0-0/MMA-Net) dataset.
    
### Directory structure                  
    ├── OpenLane                # dataset directory
    │   ├── images              # Original images
    │   ├── OpenLane-V
    |   |   ├── label           # lane labels formatted into pickle files
    |   |   ├── list            # training/test video datalists
    ├── VIL-100
    │   ├── JPEGImages          # Original images
    │   ├── Annotations         # We do not use this directory
    |   └── ...
    
### Train and Test
1. For OpenLane-V dataset:
```
$ torchrun --nproc_per_node=4 trainOL.py
$ python testOLV3.py
```
2. For VIL-100 dataset:
```
$ torchrun --nproc_per_node=4 trainVIL.py
$ python testVIL.py
```

### Evaluation 
```
$ cd ./evaluation
$ python evaluateOL.py
$ python evaluateVIL.py
```

### Acknowledgement
* [dongkwonjin/RVLD](https://github.com/dongkwonjin/RVLD)
* [yujun0-0/MMA-Net](https://github.com/yujun0-0/MMA-Net)
* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
* [Turoad/CLRNet](https://github.com/Turoad/CLRNet)