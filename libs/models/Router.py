import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from mmcv.cnn import ConvModule

def sigmoid(logits, hard=False, threshold=0.5):
    y_soft = logits.sigmoid()
    if hard:
        indices = (y_soft < threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

class AdaptiveRouter(nn.Module):
    def __init__(self, features_channels, out_channels, reduction=4):
        super(AdaptiveRouter, self).__init__()
        self.inp = sum(features_channels)
        self.oup = out_channels
        self.reduction = reduction
        self.layer1 = nn.Conv2d(self.inp, self.inp//self.reduction, kernel_size=1, bias=True)
        self.layer2 = nn.Conv2d(self.inp//self.reduction, self.oup, kernel_size=1, bias=True)
    
    def forward(self, xs, thres=0.5):
        xs = [x.mean(dim=(2, 3), keepdim=True) for x in xs]
        xs = torch.cat(xs, dim=1)
        xs = self.layer1(xs)
        xs = F.relu(xs, inplace=True)
        xs = self.layer2(xs).flatten(1)
        if self.training:
            xs = sigmoid(xs, hard=False, threshold=thres)
        else:
            xs = xs.sigmoid()
        return xs
    
class AdaptiveRouter4Lane(nn.Module):
    def __init__(self, num_priors=240, features_channels=64, num_points = 36,
                       out_channels=1, reduction=4, stages = 3):
        super(AdaptiveRouter4Lane, self).__init__()
        # self.stages = stages
        self.inp = features_channels * num_points // reduction
        layer = nn.Sequential(nn.Linear(features_channels*num_points, self.inp),
                                   nn.ReLU(),
                                   nn.Linear(self.inp, out_channels),
                                   nn.ReLU())
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(stages)])
        # self.final_layer = nn.Linear(stages, out_channels)
        self.init_weights()

        pre_norm = nn.LayerNorm([features_channels, num_points])
        self.pre_norm = nn.ModuleList([copy.deepcopy(pre_norm) for _ in range(stages)])
        DWblock = nn.Sequential(
            nn.Conv2d(num_priors, num_priors, kernel_size=3, padding=1, groups=num_priors),
            copy.deepcopy(pre_norm),
            nn.ReLU(),
            nn.Conv2d(num_priors, num_priors, kernel_size=3, padding=1, groups=num_priors),
            copy.deepcopy(pre_norm)
        )
        DWblocks = nn.ModuleList([copy.deepcopy(DWblock) for _ in range(4)]) #两个block
        self.DWNets = nn.ModuleList([copy.deepcopy(DWblocks) for _ in range(stages)]) #stages个网络


    def init_weights(self): #需要换为其他初始化方案吗？
        for m in self.modules():
            if isinstance(m, nn.Linear):
                tanh_gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)
    
    def forward(self, xs, stage, thres=0.5): # xs [1, 240, 64, 36]
        score = self.pre_norm[stage](xs)
        for DWblock in self.DWNets[stage]:
            score = F.relu(DWblock(score) + score)
        score = self.layers[stage](score.flatten(2))
        if self.training:
            score = sigmoid(score, hard=False, threshold=thres)
        else:
            score = score.sigmoid()
        return score
    
class AdaptiveRouter4LaneV2(nn.Module):
    def __init__(self, features_channels=[32, 16, 8], num_points=[24, 48, 96], reduction=2, stages=3):
        super(AdaptiveRouter4LaneV2, self).__init__()
        self.inplane = features_channels[0] * num_points[0]
        assert self.inplane==features_channels[1]*num_points[1]
        assert self.inplane==features_channels[2]*num_points[2]

        self.layers = nn.ModuleList()
        for stage in range(stages):
            layer = nn.Sequential(
                    ConvModule(in_channels=features_channels[stage], 
                                out_channels=features_channels[stage]//reduction,
                                kernel_size=3,
                                padding=1,
                                stride=1,
                                conv_cfg=dict(type='Conv1d'),
                                norm_cfg=dict(type='BN1d')),
                    ConvModule(in_channels=features_channels[stage]//reduction,
                                out_channels=features_channels[stage]//features_channels[-1],
                                kernel_size=1,
                                padding=0,
                                conv_cfg=dict(type='Conv1d'),
                                norm_cfg=dict(type='BN1d')),
                    nn.Flatten(1),
                    nn.Linear(in_features= features_channels[stage]*num_points[stage]//features_channels[-1],
                              out_features= num_points[stage])
            )
            self.layers.append(layer)
            # pre_norm = nn.LayerNorm([features_channels[stage], num_points[stage]])
            # self.pre_norm.append(pre_norm)
        self.init_weights()

    def init_weights(self): #需要换为其他初始化方案吗？
        for m in self.modules():
            if isinstance(m, nn.Linear):
                tanh_gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)
    
    def forward(self, xs, stage, thres=0.5): # xs:[1, 240, C, Length]
        # input = xs.transpose(2,3).flatten(2) #[1, 240, 768]
        B, N, C, P = xs.shape
        input = xs.reshape(B*N, C, P)
        score = self.layers[stage](input).reshape(B, N, -1)
        #我感觉能学到num_points[stage]个点的置信度均值
        score = torch.mean(score, dim=-1, keepdim=True) 
        if self.training:
            score = sigmoid(score, hard=False, threshold=thres)
        else:
            score = score.sigmoid()
        return score

class AdaptiveRouter4LaneV3(nn.Module):
    def __init__(self, inplane=256, out_channels=1, reduction=4, stages=3):
        super(AdaptiveRouter4LaneV3, self).__init__()
        self.inplane = inplane
        self.layers = nn.ModuleList()
        for stage in range(stages):
            layer = nn.Sequential(nn.Linear(self.inplane, self.inplane),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(self.inplane, self.inplane//reduction),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(self.inplane//reduction, out_channels))
            self.layers.append(layer)
    def forward(self, xs, stage, thres=0.5):
        B, N, C = xs.shape
        score = self.layers[stage](xs)
        if self.training:
            score = sigmoid(score, hard=False, threshold=thres)
        else:
            score = score.sigmoid()
        return score
        