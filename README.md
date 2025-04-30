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
3. Install NMS ops
```
$ cd ./libs/ops
$ python setup.py build develop
```
4. Install  evaluation tools:
```
sudo ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2
cd evaluation/culane
mkdir build && cd build
cmake ..
make
mv culane_evaluator ../
```

Pytorch can be installed on [here](https://pytorch.org/get-started/previous-versions/). Other versions might be available as well.

### Dataset
1. Download [OpenLane-V](https://github.com/dongkwonjin/RVLD) and [VIL-100](https://github.com/yujun0-0/MMA-Net) dataset.
2. Unzip and move dataset into ROOT/dataset
#### Directory structure of dataset          
    ├── OpenLane                
    │   ├── images              
    │   ├── OpenLane-V
    |   |   ├── label          
    |   |   ├── list            
    ├── VIL-100
    │   ├── JPEGImages          
    │   ├── Annotations
    |   └── ...
    
### Train and Test
0. Modify the settings in option files in `./options/option*.py`
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
### Weights
Weight for OpenLane-V: [here](通过网盘分享的文件：50.pth.tar
链接: https://pan.baidu.com/s/15RTib-EKtDbYn7VFi13MHA?pwd=b4ac 提取码: b4ac).

Weight for VIL-100: [here](通过网盘分享的文件：50.pth.tar
链接: https://pan.baidu.com/s/1LBEEQMBKZb8sBeSZYOUTdg?pwd=jkba 提取码: jkba).

### Evaluation 
```
$ cd ./evaluation
$ python evaluate_iou.py #(VIL-100)
$ python evaluate_iou4OL.py #(OpenLane-V)
```

### Acknowledgement
* [dongkwonjin/RVLD](https://github.com/dongkwonjin/RVLD)
* [yujun0-0/MMA-Net](https://github.com/yujun0-0/MMA-Net)
* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
* [Turoad/CLRNet](https://github.com/Turoad/CLRNet)
