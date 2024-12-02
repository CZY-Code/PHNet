#挖掘occlusion标签后的损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utility import mask_iou
from .focal_loss import FocalLoss
from .accuracy import accuracy

from .dynamic_assign import assign, liou_loss, liou_loss_diff

def binary_entropy_loss(pred, target, num_object, eps=0.001):
    ce = - 1.0 * target * torch.log(pred + eps) - (1 - target) * torch.log(1 - pred + eps)
    loss = torch.mean(ce)
    # TODO: training with bootstrapping
    return loss

def cross_entropy_loss(pred, mask, num_object, bootstrap=0.4):
    # pred: [N x K x H x W]
    # mask: [N x K x H x W] one-hot encoded
    N, _, H, W = mask.shape
    pred = -1 * torch.log(pred)
    # loss = torch.sum(pred[:, :num_object+1] * mask[:, :num_object+1])
    # loss = loss / (H * W * N)
    # bootstrap
    num = int(H * W * bootstrap)
    mask = F.interpolate(mask, size=pred.shape[-2:])
    loss = torch.sum(pred[:, :num_object+1] * mask[:, :num_object+1], dim=1).view(N, -1)
    mloss, _ = torch.sort(loss, dim=-1, descending=True)
    loss = torch.mean(mloss[:, :num])
    return loss

def mask_iou_loss(pred, mask, num_object):
    N, K, H, W = mask.shape
    loss = torch.zeros(1).to(pred.device)
    start = 0 if K == num_object else 1
    mask = F.interpolate(mask, size=pred.shape[-2:])
    for i in range(N):
        loss += (1.0 - mask_iou(pred[i, start:num_object+start], mask[i, start:num_object+start]))
    loss = loss / N
    return loss

def dice_with_iou(pred, targets, gt_idx, num_object):
        targets = F.interpolate(targets, size=pred.shape[-2:], mode='bilinear', align_corners=False)
        #拉平
        target_argmax = targets.softmax(1).argmax(1)
        
        target_masks = targets.flatten(1)
        src_masks = pred.flatten(1)

        # loss_mask = F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean')
        loss_mask = F.cross_entropy(pred, target_argmax, reduction='mean')
        loss_dice = dice_loss(src_masks, target_masks) / num_object
        loss = 7 * loss_mask + 3 * loss_dice
        return loss

def dice_loss(pred, targets, num_objects, reduction='mean'): #需要先拉平
    if pred.dim() != 2 or targets.dim != 2:
        pred = pred.flatten(1)
        targets = targets.flatten(1)
        
    inputs = pred.sigmoid()
    assert inputs.shape == targets.shape
    numerator = 2 * (inputs * targets).sum(1)
    denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
    loss = 1 - (numerator) / (denominator + 1e-4)
    if reduction == 'none':
        return loss
    return loss.sum()/num_objects


class DILaneCriterionV3(nn.Module):
    def __init__(self, cfg, train_router = False, isOneStep=False,):
        super().__init__()
        num_points = cfg.num_points
        max_lanes = cfg.max_lanes
        # self.train_stage = 1 #TODO
        self.refine_layers = 3
        self.img_h = cfg.img_h
        self.img_w = cfg.img_w
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_classes = max_lanes + 1
        weights = torch.ones(self.num_classes)
        weights[0] = cfg.bg_weight
        self.ignore_label = cfg.ignore_label
        self.criterion = nn.NLLLoss(ignore_index=self.ignore_label, weight=weights)
        self.flow_criterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        self.training_router = train_router
        self.isOneStep = isOneStep
        self.occ_criterion = nn.L1Loss(reduction='sum')
        self.cls_criterion = FocalLoss(alpha=0.5, gamma=2.) #0.25 
        # self.cls_criterion = FocalLoss(alpha=[0.1, 0.9], gamma=2., ignore=False) #alpha:[0.1, 0.9] gamma:2

    def forward(self, output, gt_mask, gt_lane, gt_idx, num_objects, gt_flow, 
                diff = None, occlusion:torch.Tensor=None):
        seg_label = gt_mask.argmax(1)
        batch = {'lane_line': gt_lane,
                 'seg': seg_label}
        if self.isOneStep: #进这个
            return self.loss4OneStep(output, batch, diff, occlusion)
        elif self.training_router:
            return self.diff_loss(output, batch, diff, occlusion) #训练router时候调用
        else:
            flow_loss = self.flow_loss(output['flow'], gt_flow)
            return self.loss(output, batch), flow_loss
        
    def loss(self, output, batch,
             cls_loss_weight=2.0, #2
             yxt_loss_weight=0.5, #0.5
             iou_loss_weight=2.0, #2
             seg_loss_weight=0.5): #0

        targets = batch['lane_line'].clone()
        matched_indices, cls_loss, reg_yxtl_loss, iou_loss, cls_acc = self.line_loss(output['predictions_lists'], targets)
        if 'predictions_sec' in output:
            matched_indices_sec, cls_loss_sec, reg_yxtl_loss_sec, iou_loss_sec, cls_acc_sec = self.line_loss(output['predictions_sec'], targets)
            #stage数量对齐，避免数量级上的差异
            # cls_loss += cls_loss_sec*(len(output['predictions_lists'])/len(output['predictions_sec']))
            # reg_yxtl_loss += reg_yxtl_loss_sec*(len(output['predictions_lists'])/len(output['predictions_sec']))
            # iou_loss += iou_loss_sec*(len(output['predictions_lists'])/len(output['predictions_sec']))
            cls_loss += cls_loss_sec
            reg_yxtl_loss += reg_yxtl_loss_sec
            iou_loss += iou_loss_sec
            
        # extra segmentation loss
        seg_loss = self.criterion(F.log_softmax(output['seg'], dim=1), batch['seg'].long())

        cls_loss /= (len(targets) * self.refine_layers)
        reg_yxtl_loss /= (len(targets) * self.refine_layers)
        iou_loss /= (len(targets) * self.refine_layers)

        loss = cls_loss * cls_loss_weight + reg_yxtl_loss * yxt_loss_weight \
             + iou_loss * iou_loss_weight + seg_loss * seg_loss_weight

        return_value = {
            'loss': loss,
            'loss_stats': {
                'loss': loss,
                'cls_loss': cls_loss * cls_loss_weight,
                'reg_yxtl_loss': reg_yxtl_loss * yxt_loss_weight,
                'seg_loss': seg_loss * seg_loss_weight,
                'iou_loss': iou_loss * iou_loss_weight
            }
        }

        for i in range(self.refine_layers-2):
            return_value['loss_stats']['stage_{}_acc'.format(i)] = cls_acc[i]
        # 这里返回哪一个matched_indices还有待商榷，
        # 第一个matched表示在当前帧中有视觉线索的正样本，
        # 如果本就是通过检索过去帧得到的正样本当做memory，保存其本就被遮挡的特征没有意义？
        return loss, matched_indices 

    def flow_loss(self, pred_flows, gt_flows):
        # preds = torch.cat(pred_flows, dim=0)
        preds = pred_flows
        B, C, H, W = preds.size()
        gt_flows = gt_flows.permute(0, 3, 1, 2)
        labels = F.interpolate(gt_flows, size=(H, W), mode='bilinear', align_corners=False)
        flow_loss = torch.norm((preds - labels), p=2, dim=1).mean()
        # flow_loss = self.flow_criterion(preds, labels)
        return flow_loss
    
    def line_loss(self, predictions_lists, targets):
        cls_loss = 0
        reg_yxtl_loss = 0
        iou_loss = 0
        matched_indices = list()
        cls_criterion = FocalLoss(alpha=0.5, gamma=2.) #0.25 2
        cls_acc = []
        cls_acc_stage = []
        for stage in range(len(predictions_lists)): 
            predictions_list = predictions_lists[stage]
            for predictions, target in zip(predictions_list, targets): #batch内循环 但是batch为1
                target = target[target[:, 1] == 1]
                if len(target) == 0:
                    # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss = cls_loss + cls_criterion(cls_pred, cls_target).sum()
                    matched_indices.append([])
                    continue

                with torch.no_grad():
                    matched_row_inds, matched_col_inds = assign(predictions, target, self.img_w, self.img_h)
                    assert matched_row_inds.shape[0]-matched_col_inds.shape[0]==0 #增大num_priors并使用one-to-many的方式?
                    matched_indices.append(matched_row_inds)

                # classification targets
                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row_inds] = 1
                cls_pred = predictions[:, :2]
                cls_loss = cls_loss + cls_criterion(cls_pred, cls_target).sum() / target.shape[0]

                # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
                reg_yxtl = predictions[matched_row_inds, 2:6]
                reg_yxtl[:, 0] *= self.n_strips # start_y
                reg_yxtl[:, 1] *= (self.img_w - 1) # start_x
                reg_yxtl[:, 2] *= 180 # theta
                reg_yxtl[:, 3] *= self.n_strips # length
                target_yxtl = target[matched_col_inds, 2:6].clone()
                with torch.no_grad():
                    # ensure the predictions starts is valid
                    predictions_starts = torch.clamp((predictions[matched_row_inds, 2] * self.n_strips).round().long(), 0, self.n_strips)
                    target_starts = (target[matched_col_inds, 2] * self.n_strips).round().long()
                    target_yxtl[:, -1] -= (predictions_starts - target_starts) # reg length
                target_yxtl[:, 0] *= self.n_strips
                target_yxtl[:, 1] *= (self.img_w-1)
                target_yxtl[:, 2] *= 180
                target_yxtl[:, 3] *= self.n_strips
                reg_yxtl_loss = reg_yxtl_loss + F.smooth_l1_loss(reg_yxtl, target_yxtl, reduction='none').mean()
                
                # regression targets -> S coordinates (all transformed to absolute values)
                reg_pred = predictions[matched_row_inds, 6:]
                reg_pred *= (self.img_w - 1)
                reg_targets = target[matched_col_inds, 6:].clone()
                iou_loss = iou_loss + liou_loss(reg_pred, reg_targets, self.img_w, length=15)

                # calculate acc
                cls_accuracy = accuracy(cls_pred, cls_target)
                cls_acc_stage.append(cls_accuracy)
            cls_acc.append(sum(cls_acc_stage) / (len(cls_acc_stage)+1)) #因为一个batch只包含一张图，因此没有lane的可能性较大导致除零错误
        
        return matched_indices, cls_loss, reg_yxtl_loss, iou_loss, cls_acc
    
    def line_loss_diff(self, predictions_lists, targets): #并行计算240个candidate的损失
        cls_loss = 0
        reg_yxtl_loss = 0
        iou_loss = 0
        matched_indices = list()
        
        for stage in range(len(predictions_lists)): #3 
            predictions_list = predictions_lists[stage]
            for predictions, target in zip(predictions_list, targets): #batch内循环 但是batch为1
                target = target[target[:, 1] == 1]
                if len(target) == 0:
                    # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_target)
                    matched_indices.append([])
                    continue

                with torch.no_grad():
                    matched_row_inds, matched_col_inds = assign(predictions, target, self.img_w, self.img_h)
                    assert matched_row_inds.shape[0] == matched_col_inds.shape[0] #增大num_priors并使用one-to-many的方式?
                    matched_indices.append(matched_row_inds)    

                # classification targets
                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row_inds] = 1
                cls_pred = predictions[:, :2]
                #这里为什要除以target.shape[0]呢？
                cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_target) #/ target.shape[0] #[240]

                # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
                reg_yxtl = predictions[matched_row_inds, 2:6]
                reg_yxtl[:, 0] *= self.n_strips # start_y
                reg_yxtl[:, 1] *= (self.img_w - 1) # start_x
                reg_yxtl[:, 2] *= 180 # theta
                reg_yxtl[:, 3] *= self.n_strips # length

                target_yxtl = target[matched_col_inds, 2:6].clone()
                target_yxtl[:, 0] *= self.n_strips
                target_yxtl[:, 1] *= (self.img_w-1)
                target_yxtl[:, 2] *= 180
                target_yxtl[:, 3] *= self.n_strips

                with torch.no_grad(): #起始端对齐
                    # ensure the predictions starts is valid
                    predictions_starts = torch.clamp((predictions[matched_row_inds, 2] * self.n_strips).round().long(), 0, self.n_strips)
                    target_starts = (target[matched_col_inds, 2] * self.n_strips).round().long()
                    target_yxtl[:, -1] -= (predictions_starts - target_starts) # reg length

                reg_yxtl_loss = reg_yxtl_loss + F.smooth_l1_loss(reg_yxtl, target_yxtl, reduction='none').mean(-1) / target.shape[0] #[match_n]
                # regression targets -> S coordinates (all transformed to absolute values)
                reg_pred = predictions[matched_row_inds, 6:]
                reg_pred *= (self.img_w - 1)
                reg_targets = target[matched_col_inds, 6:].clone()
                iou_loss = iou_loss + liou_loss_diff(reg_pred, reg_targets, self.img_w, length=15) / target.shape[0] #[match_n]

        cls_loss /= (len(targets) * self.refine_layers)
        reg_yxtl_loss /= (len(targets) * self.refine_layers)
        iou_loss /= (len(targets) * self.refine_layers)
        
        return matched_indices, cls_loss, reg_yxtl_loss, iou_loss
    
    # Calculate the loss for each instance
    def CalculateInstLoss(self, matched_row, cls_loss, reg_yxtl_loss, iou_loss,
                                cls_loss_weight, yxt_loss_weight, iou_loss_weight):
        instLoss = cls_loss * cls_loss_weight
        if len(matched_row) == 0:
            return instLoss
        else:
            instLoss[matched_row] += reg_yxtl_loss * yxt_loss_weight + iou_loss * iou_loss_weight
            return instLoss

    def diff_loss(self, output, batch,
                  diff = None,
                  occlusion = None,
                  cls_loss_weight=1.5,
                  yxt_loss_weight=1.0,
                  iou_loss_weight=2.5):
        assert diff is not None

        targets = batch['lane_line'].clone()
        #感觉这里还有一定问题，应该用最后一列预测值来计算损失，同时如何进行match也是还没思考的问题
        with torch.no_grad():
            matched_fir, cls_loss_A, reg_yxtl_loss_A, iou_loss_A = self.line_loss_diff(output['predictions_lists'], targets)
            matched_sec, cls_loss_B, reg_yxtl_loss_B, iou_loss_B = self.line_loss_diff(output['predictions_sec'], targets)
        
        if occlusion is not None:
            return matched_sec, self.occlusion_loss(diff, occlusion, matched_sec, targets)
        else:
            with torch.no_grad():
                loss_A = self.CalculateInstLoss(matched_fir[-1], cls_loss_A, reg_yxtl_loss_A, iou_loss_A,
                                                cls_loss_weight, yxt_loss_weight, iou_loss_weight)
                loss_B = self.CalculateInstLoss(matched_sec[-1], cls_loss_B, reg_yxtl_loss_B, iou_loss_B,
                                                cls_loss_weight, yxt_loss_weight, iou_loss_weight)
            delta_loss = loss_A - loss_B #[240]
            loss_A = loss_A - delta_loss/2
            loss_B = loss_B + delta_loss/2
            diffOneFrame = diff[-1][0, :, 0] #只选择最后一个stage预测的难度系数进行回归
            adaptive_loss = ((1-diffOneFrame) * loss_A + diffOneFrame * loss_B).sum()
            return matched_sec, adaptive_loss
    
    def occlusion_loss(self, diffs, occlusion, matched_indices, targets):
        #这里的损失函数可能不太好用,应该找一个更加关注与异常值的损失函数
        #或者将router改为三分类网络,使用FocalLoss? 这样就必须训练时三分类,测试时二分类?
        #之前只用了diff的一个stage做参数回归，而这个函数用了三层进行回归？
        occ_loss = 0
        for diff, matched_row_inds, lanes in zip(diffs, matched_indices, targets):
            diff = diff[0, :, 0] #[240]
            #注意torch.Tensor和torch.tensor的区别
            with torch.no_grad():
                target = torch.full((diff.shape), 0.5).to(occlusion.device)
                vaild_idx = lanes[:, 1] == 1
                occ = torch.where(vaild_idx, occlusion.float()[0], 
                                    torch.tensor([0.5]).float().to(occlusion.device))
                target[matched_row_inds] = occ[vaild_idx]
            occ_loss = occ_loss + self.occ_criterion(diff, target)
        return occ_loss

    def loss4OneStep(self, output, batch, 
                     diff = None,
                     occlusion = None,
                     cls_loss_weight=2.5, #2.0
                     yxt_loss_weight=0.5, #0.5
                     iou_loss_weight=2.0): #2.0
        assert diff is not None
        targets = batch['lane_line'].clone()
        #感觉这里还有一定问题，应该用最后一列预测值来计算损失，同时如何进行match也是还没思考的问题
        matched_fir, cls_loss_A, reg_yxtl_loss_A, iou_loss_A = self.line_loss_diff(output['predictions_lists'], targets)
        matched_sec, cls_loss_B, reg_yxtl_loss_B, iou_loss_B = self.line_loss_diff(output['predictions_sec'], targets)
        loss_A = self.CalculateInstLoss(matched_fir[-1], cls_loss_A, reg_yxtl_loss_A, iou_loss_A,
                                        cls_loss_weight, yxt_loss_weight, iou_loss_weight) #[240]
        loss_B = self.CalculateInstLoss(matched_sec[-1], cls_loss_B, reg_yxtl_loss_B, iou_loss_B,
                                        cls_loss_weight, yxt_loss_weight, iou_loss_weight) #[240]
        if occlusion is not None and False:
            occ_loss = self.occlusion_loss(diff, occlusion, matched_sec, targets)
            total_loss = loss_A.sum() + loss_B.sum() + occ_loss
            return matched_sec, total_loss
        else:
            diffOneFrame = torch.stack([d for d in diff], dim=0).squeeze().mean(dim=0, keepdim=False) #[240]
            delta_loss = torch.median(loss_A - loss_B).detach() #[240]
            loss_A = loss_A - delta_loss/2
            loss_B = loss_B + delta_loss/2
            total_loss = torch.sum((1-diffOneFrame)*loss_A + diffOneFrame * loss_B)
            return matched_sec, total_loss
