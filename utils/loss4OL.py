#挖掘occlusion标签后的损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utility import mask_iou
from .focal_loss import FocalLoss
from .dynamic_assign import assign, liou_loss_diff, assignV2

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


class Criterion4OL(nn.Module):
    def __init__(self, cfg):
        super(Criterion4OL, self).__init__()
        self.refine_layers = 3
        self.img_h = cfg.img_h
        self.img_w = cfg.img_w
        self.n_strips = cfg.num_points - 1
        self.n_offsets = cfg.num_points
        self.num_classes = cfg.max_lanes + 1
        # self.cls_criterion = FocalLoss(alpha=0.9, gamma=2., ignore=True) #0.5 2
        self.cls_criterion = FocalLoss(alpha=[0.1, 0.9], gamma=2., ignore=False) #alpha:[0.1, 0.9] gamma:2
        self.stageWeight = [0.5, 1.0, 1.5]
        self.cls_weight = cfg.cls_weight
        self.reg_weight = cfg.reg_weight
        self.iou_weight = cfg.iou_weight

    def forward(self, output, gt_lane, diff=None):
        batch = {'lane_line': gt_lane}
        return self.loss4OneStep(output, batch, diff)
        
    def line_loss_diff(self, predictions_lists, targets, stageMode=False): #并行计算240个candidate的损失
        if stageMode:
            cls_loss_stage = list()
            reg_yxtl_loss_stage = list()
            iou_loss_stage = list()
        else:
            cls_loss = 0
            reg_yxtl_loss = 0
            iou_loss = 0
        matched_indices = list()
        for stage in range(len(predictions_lists)): #3 
            predictions_list = predictions_lists[stage]
            if stageMode:
                cls_loss = 0
                reg_yxtl_loss = 0
                iou_loss = 0
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
                    # matched_row_inds, matched_col_inds = anc_assign(predictions, target, self.img_w, self.img_h)
                    assert matched_row_inds.shape[0] - matched_col_inds.shape[0] == 0 #增大num_priors并使用one-to-many的方式?
                    matched_indices.append(matched_row_inds)

                # classification targets
                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row_inds] = 1
                cls_pred = predictions[:, :2]
                cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_target) #/ target.shape[0] #[240]
                # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
                reg_yxtl = predictions[matched_row_inds, 2:6]
                reg_yxtl[:, 0] *= self.n_strips # start_y
                reg_yxtl[:, 1] *= (self.img_w - 1) # start_x
                reg_yxtl[:, 2] *= 180 # theta
                reg_yxtl[:, 3] *= self.n_strips # length
                target_yxtl = target[matched_col_inds, 2:6].clone()

                # with torch.no_grad():
                #     # ensure the predictions starts is valid
                #     predictions_starts = torch.clamp((predictions[matched_row_inds, 2] * self.n_strips).round().long(), 0, self.n_strips)
                #     target_starts = (target[matched_col_inds, 2] * self.n_strips).round().long()
                #     target_yxtl[:, -1] -= (predictions_starts - target_starts) # reg length

                target_yxtl[:, 0] *= self.n_strips
                target_yxtl[:, 1] *= (self.img_w-1)
                target_yxtl[:, 2] *= 180
                target_yxtl[:, 3] *= self.n_strips
                reg_yxtl_loss = reg_yxtl_loss + F.smooth_l1_loss(reg_yxtl, target_yxtl, reduction='none').mean(-1) / len(matched_row_inds) #target.shape[0] #[match_n]
                # regression targets -> S coordinates (all transformed to absolute values)
                reg_pred = predictions[matched_row_inds, 6:]
                reg_pred *= (self.img_w - 1)
                reg_targets = target[matched_col_inds, 6:].clone()
                iou_loss = iou_loss + liou_loss_diff(reg_pred, reg_targets, self.img_w, length=15) / len(matched_row_inds) #target.shape[0] #[match_n]
            if stageMode:
                cls_loss_stage.append(cls_loss/len(targets))
                reg_yxtl_loss_stage.append(reg_yxtl_loss/len(targets))
                iou_loss_stage.append(iou_loss/len(targets))

        if stageMode:
            return matched_indices, cls_loss_stage, reg_yxtl_loss_stage, iou_loss_stage
        else:
            cls_loss /= (len(targets) * self.refine_layers)
            reg_yxtl_loss /= (len(targets) * self.refine_layers)
            iou_loss /= (len(targets) * self.refine_layers)
            return matched_indices, cls_loss, reg_yxtl_loss, iou_loss
    
    # Calculate the loss for each instance
    def CalculateInstLoss(self, matched_row, cls_loss, reg_yxtl_loss, iou_loss):
        instLoss = cls_loss * self.cls_weight
        if len(matched_row) == 0:
            return instLoss
        else:
            instLoss[matched_row] += reg_yxtl_loss * self.reg_weight + iou_loss * self.iou_weight
            return instLoss

    def loss4OneStep(self, output, batch, 
                     diff = None,
                     stageMode = False):  #False
        assert diff is not None
        targets = batch['lane_line'].clone()
        #感觉这里还有一定问题，应该用最后一列预测值来计算损失，同时如何进行match也是还没思考的问题
        matched_fir, cls_loss_A, reg_yxtl_loss_A, iou_loss_A = self.line_loss_diff(output['predictions_fir'], targets, stageMode)
        matched_sec, cls_loss_B, reg_yxtl_loss_B, iou_loss_B = self.line_loss_diff(output['predictions_sec'], targets, stageMode)
        if stageMode:
            total_loss = 0
            for stage in range(self.refine_layers):
                loss_A = self.CalculateInstLoss(matched_fir[stage], cls_loss_A[stage], reg_yxtl_loss_A[stage], iou_loss_A[stage]) #[240]
                loss_B = self.CalculateInstLoss(matched_sec[stage], cls_loss_B[stage], reg_yxtl_loss_B[stage], iou_loss_B[stage]) #[240]
                diffOneFrame = diff[stage].squeeze() #[240]
                delta_loss = torch.median(loss_A - loss_B).detach() #[240] 这里是不是可以试一下平均数？
                loss_A = loss_A - delta_loss/2
                loss_B = loss_B + delta_loss/2
                stageTotalLoss = torch.sum((1-diffOneFrame)*loss_A + diffOneFrame*loss_B)
                total_loss = total_loss + stageTotalLoss * self.stageWeight[stage]
            total_loss /= self.refine_layers
        else:
            loss_A = self.CalculateInstLoss(matched_fir[-1], cls_loss_A, reg_yxtl_loss_A, iou_loss_A) #[240]
            loss_B = self.CalculateInstLoss(matched_sec[-1], cls_loss_B, reg_yxtl_loss_B, iou_loss_B) #[240]

            diffOneFrame = torch.stack([d for d in diff], dim=0).squeeze().mean(dim=0, keepdim=False) #[240]
            delta_loss = torch.median(loss_A - loss_B).detach() #[240]
            loss_A = loss_A - delta_loss/2
            loss_B = loss_B + delta_loss/2
            total_loss = torch.sum((1-diffOneFrame)*loss_A + diffOneFrame*loss_B)
            
            # delta_loss = (loss_A-loss_B).detach()
            # loss_median = torch.median(delta_loss)
            # delta_loss = delta_loss - loss_median
            # total_loss = torch.sum(loss_A + loss_B + (1-diffOneFrame)*delta_loss)

        # delta_loss = torch.median(loss_A - loss_B) #[240]
        # loss_A4R = (loss_A - delta_loss/2).detach()
        # loss_B4R = (loss_B + delta_loss/2).detach()
        # adaptiveLoss = torch.sum((1-diffOneFrame)*loss_A4R + diffOneFrame*loss_B4R)
        # detLoss = torch.sum(loss_A + loss_B)
        # total_loss = adaptiveLoss + detLoss

        # delta_loss = torch.median(loss_A - loss_B) #[240]
        # loss_A4R = (loss_A - delta_loss/2).detach()
        # loss_B4R = (loss_B + delta_loss/2).detach()
        # adaptiveLoss = torch.sum((1-diffOneFrame)*loss_A4R + diffOneFrame*loss_B4R)
        # diffMedian = torch.median(diffOneFrame)
        # loss = torch.where(diffOneFrame>=diffMedian, loss_B, loss_A)
        # total_loss = torch.sum(loss) + adaptiveLoss

        # delta_loss = (loss_A - loss_B).detach()
        # medianLoss = torch.median(loss_A - loss_B) #[240]
        # loss_A4R = loss_A.detach()
        # loss_B4R = loss_B.detach()
        # adaptiveLoss = torch.where(delta_loss>=medianLoss, 
        #                            (1-diffOneFrame)*loss_A4R, 
        #                            diffOneFrame*loss_B4R)
        # diffOneFrame4Det = diffOneFrame.detach()
        # detLoss = (1-diffOneFrame4Det)*loss_A + diffOneFrame4Det*loss_B
        # total_loss = torch.sum(detLoss) + torch.sum(adaptiveLoss)

        return matched_sec, total_loss
