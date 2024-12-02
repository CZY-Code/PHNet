import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# from mmcv.ops.deform_conv import DeformConv2dPack
# from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack

class featModel(nn.Module):
    def __init__(self, phase='train'):
        super(featModel, self).__init__()
        self.phase = phase
        self.net = torch.nn.Sequential(*list(resnet18(pretrained=True).children())[:-2])

    def forward(self, x):
        output = self.net(x)
        return output

    def load_param(self, weight):
        s = self.state_dict()
        for key, val in weight.items():
            # process ckpt from parallel module
            if key[:6] == 'module':
                key = key[7:]
            if key in s and s[key].shape == val.shape:
                s[key][...] = val
            elif key not in s:
                print('ignore weight from not found key {}'.format(key))
            else:
                print('ignore weight of mistached shape in key {}'.format(key))
        self.load_state_dict(s)

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class warpModel(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3):
        super(warpModel, self).__init__()
        self.conv_l = nn.Conv2d(inplane, outplane, kernel_size=1)
        self.conv_c = nn.Conv2d(inplane, outplane, kernel_size=1)
        # self.flow_make = DeformConv2dPack(outplane*2, 2, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.flow_bn = FrozenBatchNorm2d(2) #nn.BatchNorm2d(2) 
        # self.flow_ac = nn.Tanh()

        self.conv = nn.Conv2d(outplane*2, outplane, kernel_size=1, padding=0, bias=False)
        # self.conv = DeformConv2dPack(outplane*2, outplane, kernel_size=1, padding=0, bias=False)
        self.bn = FrozenBatchNorm2d(outplane) #nn.BatchNorm2d(outplane) 
        self.relu = nn.ReLU(inplace=True)
    
    def generate_flow(self, last_feature, curr_feature):
        out_h, out_w = last_feature.size()[2:]
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(last_feature).to(last_feature.device)
        last_f = self.conv_l(last_feature)
        curr_f = self.conv_c(curr_feature)
        flow = self.flow_make(torch.cat([curr_f, last_f], dim=1))
        flow = self.flow_bn(flow) #

        flow = flow.permute(0, 2, 3, 1) / norm #归一化到[-1, 1]
        flow = flow.permute(0, 3, 1, 2) #[B, 2， H， W]
        return flow

    def residual_block(self, wraped_feat, curr_feat):
        output = torch.cat([curr_feat, wraped_feat], dim=1)
        output = self.relu(self.bn(self.conv(output)))
        return output

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, *_ = input.size()
        # norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) #/ norm

        output = F.grid_sample(input, grid, mode='bilinear', align_corners=True)
        return output

    def forward(self, last_feature, curr_feature, flow):
        if isinstance(last_feature, (list, tuple)):
            last_feat = last_feature[-1]
            curr_feat = curr_feature[-1]
        else:
            last_feat = last_feature
            curr_feat = curr_feature
        last_feature_origin = last_feat.clone()
        h, w = last_feat.size()[2:]
        size = (h, w)
        
        # flow = self.generate_flow(last_feat, curr_feat)
        wraped_feat = self.flow_warp(last_feature_origin, flow, size=size)
        output = self.residual_block(wraped_feat, curr_feat)

        return output
    

class taskModel(nn.Module):
    def __init__(self, inplane, mdim, num_classes=9, phase='train'):
        super(taskModel, self).__init__()
        self.warp_net = warpModel(inplane, outplane=mdim, kernel_size=3)
        self.phase = phase
        self.convFM = nn.Conv2d(inplane, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, 128)
        self.RF3 = Refine(256, 128)  # 1/8 -> 1/4
        self.RF2 = Refine(128, 64)  # 1/4 -> 1
        self.pred2 = nn.Conv2d(32, num_classes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1)
        self.reduceC = nn.Conv2d(512*2, 512, kernel_size=1, padding=0, stride=1)
        self.reduceB = nn.BatchNorm2d(512)
        self.transconv_1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.transconv_2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.transconv_3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
    
    def forward(self, low_feature, h_feature):
        x = self.warp_net(h_feature, low_feature)
        inputFeat = torch.cat([x, h_feature], dim=1)
        inputFeat = self.reduceC(inputFeat)
        inputFeat = self.reduceB(inputFeat)
        r3 = self.transconv_1(inputFeat) #[1, 256, 16, 28]
        r2 = self.transconv_2(r3) #[1, 128, 32, 56]

        m4 = self.ResMM(self.convFM(x)) #[1, 128, 8, 14]
        m3 = self.RF3(r3, m4)  # [1, 128, 16, 28]
        m3 = self.conv1x1(m3)
        m2 = self.RF2(r2, m3)  #[1, 64, 32, 56]
        
        p1 = self.transconv_3(m2) #[1, 9, 64, 112]

        p2 = self.pred2(F.relu(p1))
        raise ValueError("size 应该引入超参数")
        p = F.interpolate(p2, size=(320, 640), mode='bilinear', align_corners=False)
        return p

    def load_param(self, weight):
        s = self.state_dict()
        for key, val in weight.items():
            # process ckpt from parallel module
            if key[:6] == 'module':
                key = key[7:]
            if key in s and s[key].shape == val.shape:
                s[key][...] = val
            elif key not in s:
                print('ignore weight from not found key {}'.format(key))
            else:
                print('ignore weight of mistached shape in key {}'.format(key))
        self.load_state_dict(s)