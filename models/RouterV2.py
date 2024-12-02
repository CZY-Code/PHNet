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
# from libs.models.utils.seg_decoder import SegDecoder
from libs.models.utils.transformer import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from libs.models.SeqFormer.position_encoding import PositionalEncoding
from libs.models.utils.dynamic_head import DynamicConv
# from libs.models.DFF import warpModel
from libs.models.Router import AdaptiveRouter4Lane
from libs.models.revcolV2 import revcol_tiny
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

class RouterV2(nn.Module):
    def __init__(self,
                 num_points=72,
                 prior_feat_channels=64,
                 fc_hidden_dim=64,
                 num_priors=240, #192 #240
                 num_fc=2,
                 refine_layers=3,
                 sample_points=36,
                 cfg=None):
        super(RouterV2, self).__init__()
        self.cfg = cfg
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.reg_hidden_dim = num_points * fc_hidden_dim // num_points
        self.prior_feat_channels = prior_feat_channels

        self.register_buffer(name='sample_x_indexs', tensor=(torch.linspace(
            0, 1, steps=self.sample_points, dtype=torch.float32) *
                                self.n_strips).long())
        self.register_buffer(name='prior_feat_ys', tensor=torch.flip(
            (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]))
        self.register_buffer(name='prior_ys', tensor=torch.linspace(1, 0,
                                                    steps=self.n_offsets,
                                                    dtype=torch.float32))

        self._init_prior_embeddings()
        init_priors, priors_on_featmap = self.generate_priors_from_embeddings() #None, None
        self.register_buffer(name='priors', tensor=init_priors)
        self.register_buffer(name='priors_on_featmap', tensor=priors_on_featmap)
        
        # first classification head
        reg_modules = list()
        cls_modules = list()
        for _ in range(num_fc):
            reg_modules += [*LinearModule(self.reg_hidden_dim)]
            cls_modules += [*LinearModule(self.reg_hidden_dim)]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)
        # n offsets + 1 length + start_x + start_y + theta
        self.reg_layers = nn.Linear(self.reg_hidden_dim, self.n_offsets + 1 + 2 + 1) #是否要加一个激活函数？？？
        self.cls_layers = nn.Linear(self.reg_hidden_dim, 2)

        reg_modules_sec = list()
        cls_modules_sec = list()
        for _ in range(num_fc):
            reg_modules_sec += [*LinearModule(self.reg_hidden_dim*2)]
            cls_modules_sec += [*LinearModule(self.reg_hidden_dim*2)]
            # reg_modules_sec += [*LinearModule(self.reg_hidden_dim)]
            # cls_modules_sec += [*LinearModule(self.reg_hidden_dim)]
        self.reg_modules_sec = nn.ModuleList(reg_modules_sec)
        self.cls_modules_sec = nn.ModuleList(cls_modules_sec)
        # n offsets + 1 length + start_x + start_y + theta
        self.reg_layers_sec = nn.Linear(self.reg_hidden_dim*2, self.n_offsets + 1 + 2 + 1) #是否要加一个激活函数？？？
        self.cls_layers_sec = nn.Linear(self.reg_hidden_dim*2, 2)
        # self.reg_layers_sec = nn.Linear(self.reg_hidden_dim, self.n_offsets + 1 + 2 + 1) #是否要加一个激活函数？？？
        # self.cls_layers_sec = nn.Linear(self.reg_hidden_dim, 2)

        # init the weights here
        self.init_weights()
        decoder_layer = TransformerDecoderLayer(d_model=self.reg_hidden_dim*2,
                                                nhead=8, dim_feedforward=256, #dim_feedforward
                                                dropout=0.1, activation="gelu",
                                                normalize_before=True)
        encoder_norm = nn.LayerNorm(self.reg_hidden_dim*2)
        self.transformer_Dec = TransformerDecoder(decoder_layer, 2, encoder_norm)
        self.PositionEmbedding = PositionalEncoding(d_hid=self.reg_hidden_dim, n_position = self.num_priors, 
                                                    temperature=16, normalize=True)
        
        DHead = DynamicConv(feat_size=self.sample_points, inplanes=self.reg_hidden_dim, early_return=False)
        self.DHead_series = nn.ModuleList([copy.deepcopy(DHead) for _ in range(self.refine_layers)])
        self.pro_embedding = nn.Embedding(self.num_priors, self.prior_feat_channels) #startY startX theta
        # self.pro_embedding = nn.ModuleList([copy.deepcopy(pro_embeddings) for _ in range(self.refine_layers)])
        # linear4C = nn.Sequential(nn.Linear(36, 9),
        #                                     nn.ReLU(),
        #                                     nn.Linear(9,1),
        #                                     nn.ReLU())
        # self.linear4Channel = nn.ModuleList([copy.deepcopy(linear4C) for _ in range(self.refine_layers)])
        self.router = AdaptiveRouter4Lane(num_priors=self.num_priors, 
                                          features_channels=self.prior_feat_channels, 
                                          num_points = self.sample_points,
                                          out_channels=1, reduction=4, 
                                          stages = self.refine_layers)

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

    def forward(self, x, last_cuts=None):
        curr_feat = x
        batch_features = list(curr_feat)
        batch_features.reverse() #from top to bottom
        batch_size = batch_features[-1].shape[0]
        if self.training: #训练时连通embedding进行梯度传递，验证时不进行连通！
            self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings()
        priors, priors_on_featmap = self.priors.repeat(batch_size, 1, 1), self.priors_on_featmap.repeat(
                                                batch_size, 1, 1) #[B, 192, 78] [B, 192, 36]
        # priors, priors_on_featmap = self.priors.repeat(batch_size, 1, 1).detach(), self.priors_on_featmap.repeat(
        #                                                batch_size, 1, 1).detach() #[B, 192, 78] [B, 192, 36]
        pro_feat = self.pro_embedding.weight.repeat(batch_size, 1, 1) #[1, 240, 128]

        predictions_lists = []
        predictions_sec = []
        attn_feat_list = []
        difficuly_score_list = []
        for stage in range(self.refine_layers): # 进行三次pooling
            prior_xs = torch.flip(priors_on_featmap, dims=[2])
            #[24*192, 64, 36, 1]
            batch_prior_features = self.pool_prior_features(batch_features[stage], self.num_priors, prior_xs)
            batch_prior_features = batch_prior_features.reshape(batch_size, self.num_priors, self.prior_feat_channels, self.sample_points)
            
            #-------router----------------
            difficuly_score = self.router(batch_prior_features.detach(), stage) #[B, 240, 1]
            difficuly_score_list.append(difficuly_score)

            #-----------feature enhancement------------#
            roi_feat = batch_prior_features.transpose(2, 3) #[B, 192, 36, 64] [B,A,P,C]
            decode_feat_l = self.DHead_series[stage](pro_feat, roi_feat) # proposal_feat roi_feat
            pro_feat = decode_feat_l.detach() #iterative dynamic enhance
            # decode_feat_l = self.linear4Channel[stage](batch_prior_features).view(batch_size, self.num_priors, -1)

            #------------the first head---------------#
            predictions, prediction_fir = self.forward_first(decode_feat_l, priors.clone())
            predictions_lists.append(predictions)

            #------------the second head--------------#
            contentFeat = decode_feat_l.clone().transpose(0, 1) #.detach()
            posEembed = self.PositionEmbedding(contentFeat)
            attnFeat = torch.cat([contentFeat, posEembed], dim=-1)
            attn_feat_list.append(attnFeat) #返回给后续帧使用
            # 取当前这个这个stage的过去帧正样本candidate特征
            last_cut = torch.cat([frame[stage] for frame in last_cuts], dim=0) if last_cuts is not None else None
            output_sec, prediction_sec = self.forward_second(last_cut, attnFeat, priors.clone())
            predictions_sec.append(output_sec)

            #------------prediction_lines-------------#
            head_weight = difficuly_score.detach()
            prediction_lines = (1-head_weight) * prediction_fir + head_weight * prediction_sec

            # refinement过程
            if stage != self.refine_layers - 1: 
                priors = prediction_lines.detach().clone() # 不共享内存也不叠加梯度
                priors_on_featmap = priors[..., 6 + self.sample_x_indexs]
        output = {'predictions_lists': predictions_lists, 'seg': None, 
                  'flow': None, 'predictions_sec': predictions_sec}
        
        return output, attn_feat_list, difficuly_score_list
    
    def forward_first(self, decode_feat_l, priors):
        batch_size = decode_feat_l.shape[0]
        cls_features = decode_feat_l.clone()
        reg_features = decode_feat_l.clone()
        for cls_layer in self.cls_modules:
            cls_features = cls_layer(cls_features)
        for reg_layer in self.reg_modules:
            reg_features = reg_layer(reg_features)
        cls_logits = self.cls_layers(cls_features)
        reg = self.reg_layers(reg_features)
        cls_logits = cls_logits.reshape(batch_size, self.num_priors, 2)  # (B, num_priors, 2)
        reg = reg.reshape(batch_size, self.num_priors, self.n_offsets + 1 + 2 + 1)
        # 如果是第二次进循环，那么此处的priors是由prediction_lines.detach().clone()得来的
        # prediction_lines不包含上一次回归预测的offset
        predictions = priors # 不共享内存但梯度叠加
        predictions[:, :, :2] = cls_logits
        predictions[:, :, 2:5] += torch.tanh(reg[:, :, :3])  #1 start_y, 1 start_x, 1 theta
        # predictions[:, :, 2:5] += reg[:, :, :3]
        predictions[:, :, 5] = reg[:, :, 3]  # length
        def tran_tensor(t:torch.Tensor):
            return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)
        predictions[..., 6:] = (
            tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(batch_size, self.num_priors, 1) -
            tran_tensor(predictions[..., 2])) * self.img_h /
            torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)
        # 只用了前面reg回归得到的前三个值，该值的作用就是利用reg linear层的三个预测值得到下一个循环中的priors
        prediction_lines = predictions.clone()
        #预测的offsets 每次全连接预测的offset都应该是相对于原始anchor的尺度
        predictions[..., 6:] += reg[..., 4:]
        return predictions, prediction_lines

    def forward_second(self, last_cut, attn_feat, priors):
        num_priors, batch_size, _ = attn_feat.shape
        if last_cut is not None and last_cut.shape[0] != 0:
            decode_feat_g = self.transformer_Dec(tgt=attn_feat, memory=last_cut).transpose(0, 1) #[192, 24, 64]
        else:
            # decode_feat_g = self.transformer_Dec(tgt=attn_feat, memory=attn_feat).transpose(0, 1) #[192, 24, 64]
            decode_feat_g = attn_feat
        # decode_feat = torch.cat([decode_feat_l, decode_feat_g], dim=-1) #24, 192, 128
        decode_feat = decode_feat_g
        cls_features = decode_feat.clone()
        reg_features = decode_feat.clone()
        for cls_layer in self.cls_modules_sec:
            cls_features = cls_layer(cls_features)
        for reg_layer in self.reg_modules_sec:
            reg_features = reg_layer(reg_features)
        cls_logits = self.cls_layers_sec(cls_features)
        reg = self.reg_layers_sec(reg_features)
        cls_logits = cls_logits.reshape(batch_size, self.num_priors, 2)  # (B, num_priors, 2)
        reg = reg.reshape(batch_size, self.num_priors, self.n_offsets + 1 + 2 + 1)

        # 如果是第二次进循环，那么此处的priors是由prediction_lines.detach().clone()得来的
        # prediction_lines不包含上一次回归预测的offset
        predictions = priors # 不共享内存但梯度叠加
        predictions[:, :, :2] = cls_logits
        predictions[:, :, 2:5] += torch.tanh(reg[:, :, :3])  #1 start_y, 1 start_x, 1 theta
        # predictions[:, :, 2:5] += reg[:, :, :3]
        predictions[:, :, 5] = reg[:, :, 3]  # length
        def tran_tensor(t):
            return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)
        predictions[..., 6:] = (
            tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(batch_size, num_priors, 1) -
            tran_tensor(predictions[..., 2])) * self.img_h /
            torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)
        # 只用了前面reg回归得到的前三个值，该值的作用就是利用reg的三个预测值得到下一个循环中的priors
        prediction_lines = predictions.clone()
        #预测的offsets 每次全连接预测的offset都应该是相对于原始anchor的尺度
        predictions[..., 6:] += reg[..., 4:] # 预测的offsets
        return predictions, prediction_lines

    def predictions_to_pred(self, predictions, ori_img_h, cut_height=0):
        '''
        Convert predictions to internal Lane structure for evaluation.
        '''
        self.prior_ys = self.prior_ys.to(predictions.device)
        self.prior_ys = self.prior_ys.double()
        lanes = []
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
            # lane_xs[:start] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)

            lane_ys = (lane_ys * (ori_img_h - cut_height) + cut_height) / ori_img_h #
            
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

    def get_lanes(self, output, org_size, cut_scale=0, as_lanes=True):
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
            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
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
                org_height = org_size[0]
                cut_height = int(org_height*cut_scale)
                pred = self.predictions_to_pred(predictions, org_height, cut_height)
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


class RouterWithB(nn.Module):
    def __init__(self, cfg, criterion = None):
        super(RouterWithB, self).__init__()
        if cfg.backbone == 'revcol': #这里有一定的bug
            self.backbone = revcol_tiny(save_memory=True, inter_supv=True)
        else:
            self.backbone = Encoder(cfg=cfg)
        self.router = RouterV2(cfg=cfg)
        self.criterion = criterion
        self.save_freq = cfg.save_freq
        self.save_freq_max = cfg.save_freq_max
        self.cut_scale = cfg.cut_scale
    
    def forward(self, inputs:dict()):
        # memory
        last_cuts = []
        matched_indices = None
        frame, mask, lanes, lane_ids, gt_flows, occlusion, num_objects, info = inputs.values()
        T, C, H, W = frame.size()
        curr_feats = self.backbone(frame)
        if self.training:
            total_loss = 0.0
        else:
            cilp_outputs = {'lane_lines':[]}

        for t in range(self.save_freq): #0-1 循环clip进行逐帧分析
            curr_feat = tuple(curr_feats[i][t:t+1] for i in range(len(curr_feats)))
            outputs, curr_cut, diff_list = self.router(curr_feat, last_cuts=None)
            if self.training:
                gt_mask = mask[t:t+1] #[1, 9, 320, 640]
                gt_idx = lane_ids[t:t+1]
                gt_lane = lanes[t:t+1]
                gt_flow = gt_flows[t:t+1]
                matched_indices, currframe_loss = self.criterion(outputs, gt_mask, gt_lane, gt_idx, 
                                                                 num_objects, gt_flow, diff_list, 
                                                                 occlusion[t:t + 1])
                total_loss += currframe_loss
            else:
                diffOneFrame = torch.stack([d for d in diff_list], dim=0).mean(dim=0, keepdim=False) #[240]
                lane_lines = torch.where(diffOneFrame>=0.5, outputs['predictions_sec'][-1], outputs['predictions_lists'][-1])
                lane_lines, keep_inds, keep = self.router.get_lanes(lane_lines, info['size'], self.cut_scale) #第一张图的车道线list
                cilp_outputs['lane_lines'].append(lane_lines[0])

            with torch.no_grad():
                if self.training: #refine过程中每一步都返回了一次match indices值
                    # last_cuts.append([c[matched_indices[-1]] for c in curr_cut]) 
                    last_cuts.append(self.saveMemory(matched_indices, curr_cut))
                else:
                    # last_cuts.append([c[keep_inds][keep] for c in curr_cut]) #list(list(positive))
                    last_cuts.append(self.saveMemory4Test(keep_inds, keep, curr_cut))
            
        for t in range(self.save_freq, T): #1-16 循环clip进行逐帧分析
            curr_feat = tuple(curr_feats[i][t:t+1] for i in range(len(curr_feats)))
            outputs, curr_cut, diff_list = self.router(curr_feat, last_cuts) #在过去5帧的正样本中做attention
            # output = {'predictions_lists': predictions_lists, 'seg': seg, 'flow': None, 'predictions_sec': predictions_sec}
            # vis_while_train(outputs)
            if self.training:
                gt_mask = mask[t:t + 1] #[1, 9, 320, 640]
                gt_idx = lane_ids[t:t+1] #
                gt_lane = lanes[t:t+1] #
                # visWhileTrain(gt_lane[0], outputs['predictions_lists'], frame[t])
                gt_flow = gt_flows[t:t+1] #
                matched_indices, currframe_loss = self.criterion(outputs, gt_mask, gt_lane, gt_idx, 
                                                                 num_objects, gt_flow, diff_list, 
                                                                 occlusion[t:t + 1])
                total_loss += currframe_loss
            else:
                diffOneFrame = torch.stack([d for d in diff_list], dim=0).mean(dim=0, keepdim=False) #[240]
                lane_lines = torch.where(diffOneFrame>=0.5, outputs['predictions_sec'][-1], outputs['predictions_lists'][-1])
                lane_lines, keep_inds, keep = self.router.get_lanes(lane_lines, info['size'], self.cut_scale) #每一张图的车道线list
                cilp_outputs['lane_lines'].append(lane_lines[0])

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
        for matched, currFeat in zip(matched_indices, curr_cut):
            mask = torch.zeros(self.router.num_priors, dtype=torch.bool)
            mask[matched] = True
            posFeat = currFeat[mask]
            negFeat = torch.mean(currFeat[~mask], dim=0, keepdim=True)
            memory.append(torch.cat([posFeat, negFeat], dim=0))
        return memory
    
    def saveMemory4Test(self, keep_inds, keep, curr_cut):
        memory = []
        for currFeat in curr_cut:
            mask = torch.zeros(self.router.num_priors, dtype=torch.bool).to(keep_inds.device)
            mask[keep_inds][keep] = True #有问题
            posFeat = currFeat[mask]
            negFeat = torch.mean(currFeat[~mask], dim=0, keepdim=True)
            memory.append(torch.cat([posFeat, negFeat], dim=0))
        return memory

def visWhileTrain(gt_lane, lane_lines, frame): #gt pred img
    img_h = 384
    img_w = 768
    n_offsets = 72
    from torchvision.transforms.functional import to_pil_image
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)
    img_pil = to_pil_image(frame.clone().detach().cpu()*std + mean)
    img_ary=np.array(img_pil) # 深拷贝，通道顺序是 RGB, (H,W,C)
    lanes = gt_lane.clone().detach().cpu().numpy()[:, 6:]
    ys = np.linspace(img_h-1, 0, num=n_offsets).astype('int')
    for lane in lanes:
        idx = np.where((lane>0) & (lane<img_w))
        ys_ = ys[idx]
        lane_ = lane[idx]
        print(ys_)
        print(lane_)
        if len(ys_) <= 2:
            continue
        for i in range(1, len(ys_)):
            cv2.line(img_ary, (int(lane_[i-1]), ys_[i-1]), 
                     (int(lane_[i]), ys_[i]), (128, 0, 0), thickness=2)
    cv2.imshow('img', img_ary)
    cv2.waitKey(0)