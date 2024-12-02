import math
import cv2
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from libs.utils.lane import Lane
from libs.ops import nms
from libs.models.utils.roi_gather import LinearModule
from libs.models.utils.transformer import TransformerDecoder, TransformerDecoderLayer, AttentionLayer
from libs.models.SeqFormer.position_encoding import PositionalEncoding, PositionalEncodingLearned
from libs.models.utils.dynamic_head import DynamicConv
from libs.models.Router import AdaptiveRouter4Lane, AdaptiveRouter4LaneV3
# from libs.models.revcolV2 import revcol_tiny
from libs.models.resnet import ResNetWrapper
from libs.models.fpn import FPN

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        # self.cfg = cfg
        self.backbone = ResNetWrapper(**cfg.backbone)
        self.neck = FPN(**cfg.neck) if cfg.haskey('neck') else None
    
    def forward(self, batch):
        fea = self.backbone(batch['img'] if isinstance(batch, dict) else batch)
        fea = self.neck(fea)
        return fea

'''
认为存在semantic misalignment problem,
三个stage分别对cls,reg,iou进行预测
'''
class DetNetV3(nn.Module):
    def __init__(self,
                 prior_feat_channels=64,
                 fc_hidden_dim=64,
                 num_fc=2,
                 refine_layers=3,
                 sample_points=36,
                 cfg=None):
        super(DetNetV3, self).__init__()
        self.cfg = cfg
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.n_strips = self.cfg.num_points - 1
        self.n_offsets = self.cfg.num_points
        self.num_priors = self.cfg.num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.reg_hidden_dim = fc_hidden_dim
        self.prior_feat_channels = prior_feat_channels

        self.register_buffer(name='sample_x_indexs', tensor=(torch.linspace(
            0, 1, steps=self.sample_points, dtype=torch.float32) * self.n_strips).long())
        self.register_buffer(name='prior_feat_ys', tensor=torch.flip(
            (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]))
        self.register_buffer(name='prior_ys', tensor=torch.linspace(1, 0,
                             steps=self.n_offsets, dtype=torch.float32))

        # self._init_prior_embeddingsV2()
        self._init_prior_embeddings()
        init_priors, priors_on_featmap = self.generate_priors_from_embeddings()
        self.register_buffer(name='priors', tensor=init_priors) #将self.prior_embeddings.weight存进buffer
        self.register_buffer(name='priors_on_featmap', tensor=priors_on_featmap)

        # first classification head
        reg_modules = list()
        cls_modules = list()
        iou_modules = list()
        for _ in range(num_fc):
            reg_modules += [*LinearModule(self.reg_hidden_dim)]
            cls_modules += [*LinearModule(self.reg_hidden_dim)]
            iou_modules += [*LinearModule(self.reg_hidden_dim)]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)
        self.iou_modules = nn.ModuleList(iou_modules)
        # n offsets + 1 length + start_x + start_y + theta
        self.reg_layers = nn.Linear(self.reg_hidden_dim, 1 + 2 + 1)
        self.cls_layers = nn.Linear(self.reg_hidden_dim, 2)
        self.iou_layers = nn.Linear(self.reg_hidden_dim, self.n_offsets)

        reg_modules_last = list()
        cls_modules_last = list()
        iou_modules_last = list()
        for _ in range(1):
            reg_modules_last += [*LinearModule(self.reg_hidden_dim)]
            cls_modules_last += [*LinearModule(self.reg_hidden_dim)]
            iou_modules_last += [*LinearModule(self.reg_hidden_dim)]
        self.reg_modules_last = nn.ModuleList(reg_modules_last)
        self.cls_modules_last = nn.ModuleList(cls_modules_last)
        self.iou_modules_last = nn.ModuleList(iou_modules_last)
        # n offsets + 1 length + start_x + start_y + theta
        self.reg_layers_last = nn.Linear(self.reg_hidden_dim, 1 + 2 + 1)
        self.cls_layers_last = nn.Linear(self.reg_hidden_dim, 2)
        self.iou_layers_last = nn.Linear(self.reg_hidden_dim, self.n_offsets)

        reg_modules_sec = list()
        cls_modules_sec = list()
        iou_modules_sec = list()
        for _ in range(num_fc):
            reg_modules_sec += [*LinearModule(self.reg_hidden_dim*2)]
            cls_modules_sec += [*LinearModule(self.reg_hidden_dim*2)]
            iou_modules_sec += [*LinearModule(self.reg_hidden_dim*2)]
        self.reg_modules_sec = nn.ModuleList(reg_modules_sec)
        self.cls_modules_sec = nn.ModuleList(cls_modules_sec)
        self.iou_modules_sec = nn.ModuleList(iou_modules_sec)
        # n offsets + 1 length + start_x + start_y + theta
        self.reg_layers_sec = nn.Linear(self.reg_hidden_dim*2, 1 + 2 + 1) #是否要加一个激活函数？？？
        self.cls_layers_sec = nn.Linear(self.reg_hidden_dim*2, 2)
        self.iou_layers_sec = nn.Linear(self.reg_hidden_dim*2, self.n_offsets)

        reg_modules_sec_last = list()
        cls_modules_sec_last = list()
        iou_modules_sec_last = list()
        for _ in range(num_fc):
            reg_modules_sec_last += [*LinearModule(self.reg_hidden_dim*2)]
            cls_modules_sec_last += [*LinearModule(self.reg_hidden_dim*2)]
            iou_modules_sec_last += [*LinearModule(self.reg_hidden_dim*2)]
        self.reg_modules_sec_last = nn.ModuleList(reg_modules_sec_last)
        self.cls_modules_sec_last = nn.ModuleList(cls_modules_sec_last)
        self.iou_modules_sec_last = nn.ModuleList(iou_modules_sec_last)
        # n offsets + 1 length + start_x + start_y + theta
        self.reg_layers_sec_last = nn.Linear(self.reg_hidden_dim*2, 1 + 2 + 1) #是否要加一个激活函数？？？
        self.cls_layers_sec_last = nn.Linear(self.reg_hidden_dim*2, 2)
        self.iou_layers_sec_last = nn.Linear(self.reg_hidden_dim*2, self.n_offsets)


        # init the weights here
        self.init_weights()
        encoder_norm = nn.LayerNorm(self.reg_hidden_dim * 2) #*2
        decoder_layer = TransformerDecoderLayer(d_model = self.reg_hidden_dim*2,
                                                nhead = 8, dim_feedforward = 256,
                                                dropout = 0.1, activation = "gelu", 
                                                normalize_before = True)
        self.transformer_Dec = TransformerDecoder(decoder_layer, 2, encoder_norm) #XXX考虑将一个Decoder换为3个decoder layer
        self.transformer_Dec_last = TransformerDecoder(decoder_layer, 1, encoder_norm) #XXX考虑将一个Decoder换为3个decoder layer
        # self.attention = AttentionLayer(d_model = self.reg_hidden_dim,
        #                                 nhead = 8, dim_feedforward = 256,
        #                                 dropout = 0.1, activation = "gelu", 
        #                                 normalize_before = True)

        # self.PositionEmbedding = PositionalEncoding(d_hid = self.reg_hidden_dim, n_position = self.num_priors, 
        #                                             temperature = 64, normalize = False) #64*2pi > self.num_priors
        self.PositionEmbedding = PositionalEncodingLearned(num_embeddings=self.num_priors, num_pos_feats=self.reg_hidden_dim)

        DHead = DynamicConv(feat_size=self.sample_points, inplanes=self.reg_hidden_dim, early_return=False)
        self.DHead_series = nn.ModuleList([copy.deepcopy(DHead) for _ in range(self.refine_layers)])
        self.pro_embedding = nn.Embedding(self.num_priors, self.prior_feat_channels) #startY startX theta

        # self.router = AdaptiveRouter4Lane(num_priors=self.num_priors, 
        #                                   features_channels=self.prior_feat_channels, 
        #                                   num_points = self.sample_points,
        #                                   out_channels=1, reduction=4, 
        #                                   stages = self.refine_layers)
        self.router = AdaptiveRouter4LaneV3(inplane=self.reg_hidden_dim*2, out_channels=1, reduction=4, stages=self.refine_layers)

    # function to init layer weights
    def init_weights(self):
        # initialize heads
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)
        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)
        for m in self.cls_layers_sec.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)
        for m in self.reg_layers_sec.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        '''
        pool prior feature from feature map.
        Args:
            batch_features (Tensor): Input feature maps, shape: (B, C, H, W) 
        '''
        batch_size = batch_features.shape[0]
        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1) #[24, 192, 36, 1]
        prior_ys = self.prior_feat_ys.repeat(batch_size * num_priors).view(
            batch_size, num_priors, -1, 1) #[24, 192, 36, 1]

        prior_xs = prior_xs * 2. - 1.
        prior_ys = prior_ys * 2. - 1.
        grid = torch.cat((prior_xs, prior_ys), dim=-1) #[24, 192, 36, 2]

        #output->using :attr:`input` values and pixel locations from :attr:`grid`.
        feature = F.grid_sample(batch_features, grid, align_corners=True).permute(0, 2, 1, 3) #[24, 192, 64, 36]
        feature = feature.reshape(batch_size * num_priors, self.prior_feat_channels, self.sample_points, 1)
        return feature

    def generate_priors_from_embeddings(self):
        predictions = self.prior_embeddings.weight  # (num_prop, 3) #1 start_y, 1 start_x, 1 theta
        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates, score[0] = negative prob, score[1] = positive prob
        priors = predictions.new_zeros(
            (self.num_priors, 2 + 2 + 2 + self.n_offsets), device=predictions.device)
        priors[:, 2:5] = predictions.clone() 
        priors[:, 6:] = (
            priors[:, 3].unsqueeze(1).clone().repeat(1, self.n_offsets) *
            (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(self.num_priors, 1) -
              priors[:, 2].unsqueeze(1).clone().repeat(1, self.n_offsets)) *
             self.img_h / torch.tan(priors[:, 4].unsqueeze(1).clone().repeat(
                 1, self.n_offsets) * math.pi + 1e-5))) / (self.img_w - 1)
        # init priors on feature map
        priors_on_featmap = priors.clone()[..., 6 + self.sample_x_indexs] #在72个点中均匀取36个点 不共享内存
        return priors, priors_on_featmap

    def _init_prior_embeddings(self):
        # [start_y, start_x, theta] -> all normalize

        self.prior_embeddings = nn.Embedding(self.num_priors, 3) #startY startX theta

        half_bottom_priors_nums = self.num_priors // 4 #48 一半的底部长度
        left_priors_nums = self.num_priors // 4 #48
        half_priors_nums = self.num_priors // 2 #96

        strip_size = 0.8 / (left_priors_nums // 2 - 1) #0.5 每个起始点2个anchor
        bottom_strip_size = 0.5 / (half_bottom_priors_nums // 2 + 1) #每个起始点2个anchor
        
        for i in range(left_priors_nums): #左
            nn.init.constant_(self.prior_embeddings.weight[i, 0],
                              (i // 2) * strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.16 if i % 2 == 0 else 0.32)

        for i in range(left_priors_nums, half_priors_nums): #中左
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 1],
                              ((i - left_priors_nums) // 2 + 1) *
                              bottom_strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.2 if i % 2 == 0 else 0.4) #0.2 * (i % 4 + 1))
        
        for i in range(half_priors_nums, half_priors_nums + half_bottom_priors_nums): #中右边
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 1],
                              ((i - half_priors_nums) // 2 + 1) *
                              bottom_strip_size + 0.5)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                             0.6 if i % 2 == 0 else 0.8)
        
        for i in range(half_priors_nums + half_bottom_priors_nums, self.num_priors): #右
            nn.init.constant_(
                self.prior_embeddings.weight[i, 0],
                ((i - half_priors_nums - half_bottom_priors_nums) // 2) *
                strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.68 if i % 2 == 0 else 0.84)

    def _init_prior_embeddingsV2(self):
        # [start_y, start_x, theta] -> all normalize

        self.prior_embeddings = nn.Embedding(self.num_priors, 3) #startY startX theta

        left_priors_nums = self.num_priors // 8 * 3
        bottom_left_priors_nums = self.num_priors // 8 * 2
        bottom_right_priors_nums = self.num_priors // 8 * 2
        right_priors_nums = self.num_priors // 8 * 1

        left_strip_size = 0.5 / (left_priors_nums // 2 - 1)
        right_strip_size = 0.5 / (right_priors_nums // 2 - 1)
        bottom_strip_size = 1.0 / ((bottom_left_priors_nums + bottom_right_priors_nums) // 4 + 1)
        
        for i in range(left_priors_nums): #左
            nn.init.constant_(self.prior_embeddings.weight[i, 0], i // 2 * left_strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.16 if i % 2 == 0 else 0.32)

        start_nums = left_priors_nums
        for i in range(start_nums, start_nums + bottom_left_priors_nums): #中左
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], ((i - start_nums) // 4 + 1) * bottom_strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.25 + (i % 4) * 0.05)
        
        start_nums += bottom_left_priors_nums
        for i in range(start_nums, start_nums + bottom_right_priors_nums): #中右边
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], ((i - start_nums) // 4 + 1) * bottom_strip_size + 0.5)
            nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.60 + (i % 4) * 0.05)
        
        start_nums += bottom_right_priors_nums
        for i in range(start_nums, start_nums + right_priors_nums): #右
            nn.init.constant_(self.prior_embeddings.weight[i, 0], (i - start_nums) // 2 * right_strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.68 if i % 2 == 0 else 0.84)
        
        assert self.num_priors == start_nums + right_priors_nums


    def forward(self, x, last_cuts=None, priors_last=None):
        curr_feat = x
        batch_features = list(curr_feat)
        batch_features.reverse() # from top to bottom
        batch_size = batch_features[-1].shape[0]
        if self.training: # 训练时连通embedding进行梯度传递，验证时不进行连通！
            self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings()
        priors, priors_on_featmap = self.priors.repeat(batch_size, 1, 1), self.priors_on_featmap.repeat(batch_size, 1, 1) # [B, 240, 78] [B, 240, 36]
        pro_feat = self.pro_embedding.weight.repeat(batch_size, 1, 1) # [1, 240, 128]

        predictions_fir = []
        predictions_sec = []
        attn_feat_list = []
        difficuly_score_list = []

        for stage in range(self.refine_layers): # 进行三次pooling
            prior_xs = torch.flip(priors_on_featmap, dims=[2])
            # [24*192, 64, 36, 1]
            batch_prior_features = self.pool_prior_features(batch_features[stage], self.num_priors, prior_xs)
            batch_prior_features = batch_prior_features.reshape(batch_size, self.num_priors, self.prior_feat_channels, self.sample_points)

            #-----Feature Enhancement-----
            roi_feat = batch_prior_features.transpose(2, 3) # [B, 192, 36, 64] [B,A,P,C]
            dynamicFeat = self.DHead_series[stage](pro_feat, roi_feat) # proposal_feat roi_feat
            pro_feat = dynamicFeat.detach() # iterative dynamic enhance
            # dynamicFeat = self.linear4Channel[stage](batch_prior_features).view(batch_size, self.num_priors, -1)

            contentFeat = dynamicFeat.clone().transpose(0, 1) # .detach().transpose(0, 1)
            posEembed = self.PositionEmbedding(contentFeat)
            attnFeat = torch.cat([contentFeat, posEembed], dim=-1)
            if len(last_cuts) != 0:
                memoryFeat = torch.cat([frame[stage] for frame in last_cuts], dim=0)
                memoryCont = memoryFeat[:, :, :64].transpose(0, 1)
                queryFeat = torch.cat([attnFeat, memoryFeat[:, :, :]], dim=0)
            else:
                memoryFeat = None
                memoryCont = None
                queryFeat = attnFeat
            attn_feat_list.append(queryFeat) # 返回给后续帧使用
            # last_cut = torch.cat([frame[stage] for frame in last_cuts], dim=0) if len(last_cuts) != 0 else None
                
            #------------Router-----------
            difficuly_score = self.router(queryFeat, stage).transpose(0, 1) # [B, 240, 1] batch_prior_features.detach()
            difficuly_score_list.append(difficuly_score)
            
            #------------The First Head---------------
            output_fir, priors_fir = self.forward_first(dynamicFeat, priors.clone(), stage)
            if priors_last is not None:
                output_fir_last, priors_fir_last = self.forward_first_last(memoryCont, priors_last.clone())
                output_fir = torch.concat([output_fir, output_fir_last], dim=1)
                priors_fir = torch.concat([priors_fir, priors_fir_last], dim=1)
            predictions_fir.append(output_fir)

            #------------The Second Head--------------
            # output_sec, priors_sec = self.forward_secondV2(posEembed, contentFeat, pos_last, cont_last, priors.clone(), stage)
            output_sec, priors_sec = self.forward_second(attnFeat, memoryFeat, priors.clone(), stage)
            if priors_last is not None:
                output_sec_last, priors_sec_last = self.forward_second_last(memoryFeat, attnFeat, priors_last.clone())
                output_sec = torch.concat([output_sec, output_sec_last], dim=1)
                priors_sec = torch.concat([priors_sec, priors_sec_last], dim=1)
            predictions_sec.append(output_sec)

            #-------------Refinement--------------
            if stage != self.refine_layers - 1: 
                head_weight = difficuly_score.detach()
                priors_next = (1 - head_weight) * priors_fir + head_weight * priors_sec
                priors = priors_next[:, :240, :].detach().clone() # 不共享内存也不叠加梯度
                priors_on_featmap = priors[..., 6 + self.sample_x_indexs]
                if priors_last is not None:
                    priors_last = priors_next[:, 240:, :].detach().clone()
                
        output = {'predictions_fir': predictions_fir, 'predictions_sec': predictions_sec}
        return output, attn_feat_list, difficuly_score_list
    
    def forward_first(self, decode_feat_l, predictions, stage):
        batch_size, num_priors, _ = decode_feat_l.size()

        cls_features = decode_feat_l.clone()
        for cls_layer in self.cls_modules:
            cls_features = cls_layer(cls_features)
        cls_logits = self.cls_layers(cls_features)
        cls_logits = cls_logits.reshape(batch_size, num_priors, 2)  # (B, num_priors, 2)
        predictions[:, :, :2] = cls_logits

        reg_features = decode_feat_l.clone()
        for reg_layer in self.reg_modules:
            reg_features = reg_layer(reg_features)
        reg = self.reg_layers(reg_features)
        reg = reg.reshape(batch_size, num_priors, 1 + 2 + 1)
        predictions[:, :, 2:5] += torch.tanh(reg[:, :, :3])  #1 start_y, 1 start_x, 1 theta
        # predictions[:, :, 2:5] += reg[:, :, :3]
        predictions[:, :, 5] = reg[:, :, 3]  # length

        iou_features = decode_feat_l.clone()
        for iou_layer in self.iou_modules:
            iou_features = iou_layer(iou_features)
        pred_offsets = self.iou_layers(iou_features)
        pred_offsets = pred_offsets.reshape(batch_size, num_priors, self.n_offsets)
        def tran_tensor(t:torch.Tensor):
            return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)
        predictions[..., 6:] = (
            tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(batch_size, num_priors, 1) -
            tran_tensor(predictions[..., 2])) * self.img_h /
            torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)
        prediction_lines = predictions.clone()
        predictions[..., 6:] += pred_offsets

        return predictions, prediction_lines

    def forward_first_last(self, decode_feat_l, predictions):
        batch_size, num_priors, _ = decode_feat_l.size()

        cls_features = decode_feat_l.clone()
        for cls_layer in self.cls_modules_last:
            cls_features = cls_layer(cls_features)
        cls_logits = self.cls_layers_last(cls_features)
        cls_logits = cls_logits.reshape(batch_size, num_priors, 2)  # (B, num_priors, 2)
        predictions[:, :, :2] = cls_logits

        reg_features = decode_feat_l.clone()
        for reg_layer in self.reg_modules_last:
            reg_features = reg_layer(reg_features)
        reg = self.reg_layers_last(reg_features)
        reg = reg.reshape(batch_size, num_priors, 1 + 2 + 1)
        predictions[:, :, 2:5] += torch.tanh(reg[:, :, :3])  #1 start_y, 1 start_x, 1 theta
        # predictions[:, :, 2:5] += reg[:, :, :3]
        predictions[:, :, 5] = reg[:, :, 3]  # length

        iou_features = decode_feat_l.clone()
        for iou_layer in self.iou_modules_last:
            iou_features = iou_layer(iou_features)
        pred_offsets = self.iou_layers_last(iou_features)
        pred_offsets = pred_offsets.reshape(batch_size, num_priors, self.n_offsets)
        def tran_tensor(t:torch.Tensor):
            return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)
        predictions[..., 6:] = (
            tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(batch_size, num_priors, 1) -
            tran_tensor(predictions[..., 2])) * self.img_h /
            torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)
        prediction_lines = predictions.clone()
        predictions[..., 6:] += pred_offsets

        return predictions, prediction_lines

    def forward_secondV2(self, posEembed, contentFeat, pos_last, cont_last, predictions, stage):
        num_priors, batch_size, _ = posEembed.shape
        if pos_last is not None and pos_last.shape[0] != 0:
            decode_feat = self.attention(query=posEembed, key=pos_last, value=cont_last).transpose(0, 1) #[192, 24, 64]
        else:
            # decode_feat_g = self.transformer_Dec(tgt=attn_feat, memory=attn_feat).transpose(0, 1) #[192, 24, 64]
            decode_feat = contentFeat

        cls_features = decode_feat.clone()
        for cls_layer in self.cls_modules_sec:
            cls_features = cls_layer(cls_features)
        cls_logits = self.cls_layers_sec(cls_features)
        cls_logits = cls_logits.reshape(batch_size, self.num_priors, 2)  # (B, num_priors, 2)
        predictions[:, :, :2] = cls_logits

        reg_features = decode_feat.clone()
        for reg_layer in self.reg_modules_sec:
            reg_features = reg_layer(reg_features)
        reg = self.reg_layers_sec(reg_features)
        reg = reg.reshape(batch_size, self.num_priors, 1 + 2 + 1)
        predictions[:, :, 2:5] += torch.tanh(reg[:, :, :3])  #1 start_y, 1 start_x, 1 theta
        # predictions[:, :, 2:5] += reg[:, :, :3]
        predictions[:, :, 5] = reg[:, :, 3]  # length

        iou_features = decode_feat.clone()
        for iou_layer in self.iou_modules_sec:
            iou_features = iou_layer(iou_features)
        pred_offsets = self.iou_layers_sec(iou_features)
        pred_offsets = pred_offsets.reshape(batch_size, self.num_priors, self.n_offsets)
        def tran_tensor(t):
            return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)
        predictions[..., 6:] = (
            tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(batch_size, self.num_priors, 1) -
            tran_tensor(predictions[..., 2])) * self.img_h /
            torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)
        prediction_lines = predictions.clone()
        predictions[..., 6:] += pred_offsets

        return predictions, prediction_lines

    def forward_second(self, attn_feat, last_cut, predictions, stage):
        num_priors, batch_size, _ = attn_feat.shape
        if last_cut is not None and last_cut.shape[0] != 0:
            decode_feat = self.transformer_Dec(tgt=attn_feat, memory=last_cut).transpose(0, 1) #[192, 24, 64]
        else:
            decode_feat = attn_feat

        cls_features = decode_feat.clone()
        for cls_layer in self.cls_modules_sec:
            cls_features = cls_layer(cls_features)
        cls_logits = self.cls_layers_sec(cls_features)
        cls_logits = cls_logits.reshape(batch_size, num_priors, 2)  # (B, num_priors, 2)
        predictions[:, :, :2] = cls_logits

        reg_features = decode_feat.clone()
        for reg_layer in self.reg_modules_sec:
            reg_features = reg_layer(reg_features)
        reg = self.reg_layers_sec(reg_features)
        reg = reg.reshape(batch_size, num_priors, 1 + 2 + 1)
        predictions[:, :, 2:5] += torch.tanh(reg[:, :, :3])  #1 start_y, 1 start_x, 1 theta
        # predictions[:, :, 2:5] += reg[:, :, :3]
        predictions[:, :, 5] = reg[:, :, 3]  # length

        iou_features = decode_feat.clone()
        for iou_layer in self.iou_modules_sec:
            iou_features = iou_layer(iou_features)
        pred_offsets = self.iou_layers_sec(iou_features)
        pred_offsets = pred_offsets.reshape(batch_size, num_priors, self.n_offsets)
        def tran_tensor(t):
            return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)
        predictions[..., 6:] = (
            tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(batch_size, num_priors, 1) -
            tran_tensor(predictions[..., 2])) * self.img_h /
            torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)
        prediction_lines = predictions.clone()
        predictions[..., 6:] += pred_offsets

        return predictions, prediction_lines
    
    def forward_second_last(self, queryFeat, memoryFeat, predictions):
        num_priors, batch_size, _ = queryFeat.shape
        decode_feat = self.transformer_Dec_last(tgt=queryFeat, memory=memoryFeat).transpose(0, 1) #[192, 24, 64]

        cls_features = decode_feat.clone()
        for cls_layer in self.cls_modules_sec_last:
            cls_features = cls_layer(cls_features)
        cls_logits = self.cls_layers_sec_last(cls_features)
        cls_logits = cls_logits.reshape(batch_size, num_priors, 2)  # (B, num_priors, 2)
        predictions[:, :, :2] = cls_logits

        reg_features = decode_feat.clone()
        for reg_layer in self.reg_modules_sec_last:
            reg_features = reg_layer(reg_features)
        reg = self.reg_layers_sec_last(reg_features)
        reg = reg.reshape(batch_size, num_priors, 1 + 2 + 1)
        predictions[:, :, 2:5] += torch.tanh(reg[:, :, :3])  #1 start_y, 1 start_x, 1 theta
        # predictions[:, :, 2:5] += reg[:, :, :3]
        predictions[:, :, 5] = reg[:, :, 3]  # length

        iou_features = decode_feat.clone()
        for iou_layer in self.iou_modules_sec_last:
            iou_features = iou_layer(iou_features)
        pred_offsets = self.iou_layers_sec_last(iou_features)
        pred_offsets = pred_offsets.reshape(batch_size, num_priors, self.n_offsets)
        def tran_tensor(t):
            return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)
        predictions[..., 6:] = (
            tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(batch_size, num_priors, 1) -
            tran_tensor(predictions[..., 2])) * self.img_h /
            torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)
        prediction_lines = predictions.clone()
        predictions[..., 6:] += pred_offsets

        return predictions, prediction_lines
    
    def predictions_to_pred(self, predictions, ori_img_h, cut_height=0):
        '''
        Convert predictions to internal Lane structure for evaluation.
        '''
        self.prior_ys = self.prior_ys.to(predictions.device)
        self.prior_ys = self.prior_ys.double()
        lanes = []
        #2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates
        for lane in predictions:
            lane_xs = lane[6:]  # normalized value
            start = min(max(0, int(round(lane[2].item() * self.n_strips))),
                        self.n_strips)
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)
            # end = label_end
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       ).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            
            # lane_ys = (lane_ys * (ori_img_h - cut_height) + cut_height) / ori_img_h
            
            if len(lane_xs) <= 1:
                continue
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def get_lanes(self, output, org_size, crop_size=0, as_lanes=True):
        '''
        Convert model output to lanes.
        '''
        softmax = nn.Softmax(dim=1)
        decoded = []
        for predictions in output: #循环epoch
            keep = []
            # filter out the conf lower than conf threshold
            threshold = self.cfg.test_parameters.conf_threshold
            scores = softmax(predictions[:, :2])[:, 1]
            keep_inds = scores >= threshold
            predictions = predictions[keep_inds]
            scores = scores[keep_inds]
            if predictions.shape[0] == 0:
                decoded.append([])
                continue
            nms_predictions = predictions.detach().clone()
            nms_predictions = torch.cat([nms_predictions[..., :4], nms_predictions[..., 5:]], dim=-1)
            nms_predictions[..., 3] = nms_predictions[..., 3] * (self.img_w-1)
            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips #71
            nms_predictions[..., 5:] = nms_predictions[..., 5:] * (self.img_w - 1)

            keep, num_to_keep, _ = nms(
                nms_predictions,
                scores,
                overlap=self.cfg.test_parameters.nms_thres,
                top_k=self.cfg.max_lanes)
            keep = keep[:num_to_keep]
            predictions = predictions[keep]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue
            predictions[:, 5] = torch.round(predictions[:, 5] * self.n_strips)

            if as_lanes:
                pred = self.predictions_to_pred(predictions, org_size[0], crop_size)
            else:
                pred = predictions
            decoded.append(pred)
        
        return decoded, keep_inds, keep
    
    def get_labels(self, labels, as_lanes=True):
        '''Convert labels to lanes.'''
        decoded = []
        for predictions in labels: #循环batch
            # filter out the conf lower than conf threshold
            predictions = predictions[predictions[:, 1] == 1]
            
            if predictions.shape[0] == 0:
                decoded.append([])
                continue
            predictions[:, 3] /= (self.img_w - 1) # start_x
            predictions[:, 6:] /= (self.img_w - 1) #points

            if as_lanes:
                pred = self.predictions_to_pred(predictions)
            else:
                pred = predictions
            decoded.append(pred)
        return decoded


class RouterOL(nn.Module):
    def __init__(self, cfg, criterion = None):
        super(RouterOL, self).__init__()
        self.backbone = Encoder(cfg=cfg)
        self.detNet = DetNetV3(cfg=cfg)
        self.criterion = criterion
        self.save_freq_max = cfg.save_freq_max
        self.crop_size = cfg.dscfg.crop_size
        self.org_size = (cfg.dscfg.org_height, cfg.dscfg.org_width)
    
    def forward(self, inputs:dict()):
        # memory
        last_cuts = []
        matched_indices = None
        frame, lanes = inputs.values()
        T, C, H, W = frame.size()
        curr_feats = self.backbone(frame)
        if self.training:
            total_loss = 0.0
        else:
            cilp_outputs = {'lane_lines':[]}
            last_lines = []
        priors_last = None
            
        for t in range(T): #1-16 循环clip进行逐帧分析
            curr_feat = tuple(curr_feats[i][t:t+1] for i in range(len(curr_feats)))
            outputs, curr_cut, diff_list = self.detNet(curr_feat, last_cuts, priors_last) #在过去5帧的正样本中做attention
            # vis_while_train(outputs)
            if self.training:
                gt_lane = lanes[t:t+1] #
                # visWhileTrain(gt_lane[0], frame[t]) #XXX
                matched_indices, currframe_loss, priors_last = self.criterion(outputs, gt_lane, diff_list)
                total_loss += currframe_loss
            else:
                diffOneFrame = torch.stack([d for d in diff_list], dim=0).mean(dim=0, keepdim=False) #[240]
                # diffOneFrame = diff_list[-1]
                curr_lines = torch.where(diffOneFrame>=0.5, outputs['predictions_sec'][-1], outputs['predictions_fir'][-1])
                # curr_lines = torch.concat([outputs['predictions_sec'][-1], outputs['predictions_fir'][-1]], dim=1)
                lane_lines, keep_inds, keep = self.detNet.get_lanes(curr_lines, self.org_size, self.crop_size) #每一张图的车道线list
                cilp_outputs['lane_lines'].append(lane_lines[0])
                # if len(last_lines) != 0:
                #     new_lines = torch.concat([curr_lines, last_lines], dim=1)
                #     lane_lines, _, _ = self.detNet.get_lanes(new_lines, self.org_size, self.crop_size)
                # self.visWhileTrain(lanes[t:t+1][0], frame[t])
                # self.visWhileTest(lane_lines[0], lanes[t:t+1][0], frame[t])
                priors_last = curr_lines[:, keep_inds, :][:, keep, :]

            with torch.no_grad():
                if self.training:
                    # last_cuts.append([c[matched_indices[-1]] for c in curr_cut])
                    last_cuts.append(self.saveMemory(matched_indices, curr_cut))
                else:
                    # last_cuts.append([c[keep_inds][keep] for c in curr_cut])
                    last_cuts.append(self.saveMemory4Test(keep_inds, keep, curr_cut))
                if t >= self.save_freq_max:
                    last_cuts.pop(0)
                
                    
        if self.training:
            return total_loss
        else:
            return cilp_outputs
    
    def saveMemory(self, matched_indices, curr_cut):
        memory = []
        matched = matched_indices[-1]
        for currFeat in curr_cut:
            num_priors, _, _ = currFeat.size()
            mask = torch.zeros(num_priors, dtype=torch.bool)
            mask[matched] = True
            posFeat = currFeat[mask]
            # negFeat = torch.mean(currFeat[~mask], dim=0, keepdim=True)
            # memory.append(torch.cat([posFeat, negFeat], dim=0))
            memory.append(posFeat)
        return memory
    
    def saveMemory4Test(self, keep_inds, keep, curr_cut):
        memory = []
        for currFeat in curr_cut:
            num_priors, _, _ = currFeat.size()
            mask = torch.zeros(num_priors, dtype=torch.bool).to(keep_inds.device)
            # mask[keep_inds][keep] = True #有大bug!! 
            pos_inds = torch.where(keep_inds)[0][keep]
            mask[pos_inds] = True
            posFeat = currFeat[mask]
            # negFeat = torch.mean(currFeat[~mask], dim=0, keepdim=True)
            # memory.append(torch.cat([posFeat, negFeat], dim=0))
            memory.append(posFeat)
        return memory

    def visWhileTrain(self, gt_lane, frame): #gt img
        img_h = 384
        img_w = 768
        from torchvision.transforms.functional import to_pil_image
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)
        img_pil = to_pil_image(frame.clone().detach().cpu()*std + mean)
        img_ary=np.array(img_pil) # 深拷贝，通道顺序是 RGB, (H,W,C)
        lanes = gt_lane.clone().detach().cpu().numpy()[:, 6:]
        ys = np.linspace(img_h-1, 0, num=self.detNet.n_offsets).astype('int')
        for lane in lanes:
            idx = np.where((lane>0) & (lane<img_w))
            ys_ = ys[idx]
            lane_ = lane[idx]
            # print(ys_)
            # print(lane_)
            if len(ys_) <= 2:
                continue
            for i in range(1, len(ys_)):
                cv2.line(img_ary, (int(lane_[i-1]), ys_[i-1]), 
                        (int(lane_[i]), ys_[i]), (128, 0, 0), thickness=2)
        cv2.imshow('img', img_ary)
        cv2.waitKey(0)

    def visWhileTest(self, lane_lines, gt_lane, frame): #pred img
        ori_img_h = 1280
        ori_img_h = 1920
        cut_height = 480
        img_h = 384
        img_w = 768
        from torchvision.transforms.functional import to_pil_image
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)
        img_pil = to_pil_image(frame.clone().detach().cpu()*std + mean)
        img_ary = np.array(img_pil) # 深拷贝，通道顺序是 RGB, (H,W,C)
        img_ary = cv2.cvtColor(img_ary, cv2.COLOR_BGR2RGB)

        lanes = gt_lane.clone().detach().cpu().numpy()[:, 6:]
        ys = np.linspace(img_h-1, 0, num=self.detNet.n_offsets).astype('int')
        for lane in lanes:
            idx = np.where((lane>0) & (lane<img_w))
            ys_ = ys[idx]
            lane_ = lane[idx]
            if len(ys_) <= 2:
                continue
            for i in range(1, len(ys_)):
                cv2.line(img_ary, (int(lane_[i-1]), ys_[i-1]), 
                        (int(lane_[i]), ys_[i]), (0, 128, 0), thickness=2)

        lanes = [lane.points for lane in lane_lines]
        if len(lanes) == 0:
            return
        for lane in lanes:
            if len(lane) < 2:
                continue
            x_0, y_0 = int(img_w*lane[0][0]), int(img_h*(lane[0][1]))
            for i in range(1, len(lane)):
                x_i, y_i = int(img_w*lane[i][0]), int(img_h*(lane[i][1]))
                cv2.line(img_ary, (x_0, y_0), (x_i, y_i), (128, 0, 0), thickness=2)
                x_0, y_0 = x_i, y_i

        cv2.imshow('img', img_ary)
        cv2.waitKey(0)