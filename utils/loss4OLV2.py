#挖掘occlusion标签后的损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal_loss import FocalLoss
from .dynamic_assign import assign, liou_loss, liou_loss_diff, assignV2, anc_assign, assignOne2Many, assignCrossFrame


class Criterion4OL(nn.Module):
    def __init__(self, cfg):
        super(Criterion4OL, self).__init__()
        self.img_h = cfg.img_h
        self.img_w = cfg.img_w
        self.n_strips = cfg.num_points - 1
        self.n_offsets = cfg.num_points
        self.num_classes = cfg.max_lanes + 1
        # self.cls_criterion = FocalLoss(alpha=0.9, gamma=2., ignore=True) #0.5 2
        self.cls_criterion = FocalLoss(alpha=[0.1, 0.9], gamma=2., ignore=False) #alpha:[0.1, 0.9] gamma:2
        self.cls_weight = cfg.cls_weight
        self.reg_weight = cfg.reg_weight
        self.iou_weight = cfg.iou_weight
        self.last_target = None

    def forward(self, output, gt_lane, diff=None):
        batch = {'lane_line': gt_lane}
        return self.loss4OneStep(output, batch, diff)
        
    def line_loss_diff_A(self, predictions_lists, targets): #并行计算240个candidate的损失
        cls_loss = 0
        reg_loss = 0
        iou_loss = 0
        # CrossFrameLoss = 0
        matched_indices = list()
        for stage in range(len(predictions_lists)):
            predictions_list = predictions_lists[stage]
            for predictions, target in zip(predictions_list, targets): #batch内循环 但是batch为1
                target = target[target[:, 1] == 1]
                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_pred = predictions[:, :2]
                if len(target) == 0:
                    cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_target)
                    matched_indices.append([])
                    continue
                with torch.no_grad():
                    # matched_row_inds, matched_col_inds = assign(predictions, target, self.img_w, self.img_h)
                    # matched_row_inds, matched_col_inds = anc_assign(predictions, target, self.img_w, self.img_h)
                    matched_row_inds, matched_col_inds = assignOne2Many(predictions, target, self.img_w, self.img_h)
                    assert matched_row_inds.shape[0] - matched_col_inds.shape[0] == 0
                    matched_indices.append(matched_row_inds)

                    # if self.last_target is not None:
                    #     # one-to-one match
                    #     matched_curr, matched_last = assignCrossFrame(target[matched_col_inds], self.last_target, self.img_w, self.img_h)
                    # else:
                    #     matched_curr, matched_last = None, None
                    # self.last_target = target[matched_col_inds].clone()

                # if matched_curr is not None and len(matched_curr) != 0:
                #     CrossFrameLoss += self.crossFrameLoss(matched_row_inds, matched_col_inds, matched_curr, predictions, target)

                # classification targets
                cls_target[matched_row_inds] = 1
                cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_target) # / target.shape[0] #[240]

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
                reg_loss = reg_loss + F.smooth_l1_loss(reg_yxtl, target_yxtl, reduction='none').mean()

                # regression targets -> S coordinates (all transformed to absolute values)
                reg_pred = predictions[matched_row_inds, 6:]
                reg_pred *= (self.img_w - 1)
                reg_targets = target[matched_col_inds, 6:].clone()
                iou_loss = iou_loss + liou_loss(reg_pred, reg_targets, self.img_w, length=15) #15

        cls_loss /= (len(targets) * len(predictions_lists))
        reg_loss /= (len(targets) * len(predictions_lists))
        iou_loss /= (len(targets) * len(predictions_lists))
        # CrossFrameLoss /= (len(targets) * len(predictions_lists))
        return matched_indices, cls_loss, reg_loss, iou_loss

    def line_loss_diff_B(self, predictions_lists, targets): #并行计算240个candidate的损失
        cls_loss = 0
        reg_loss = 0
        iou_loss = 0
        # CrossFrameLoss = 0

        matched_indices = list()
        for stage in range(len(predictions_lists)):
            predictions_list = predictions_lists[stage]
            for predictions, target in zip(predictions_list, targets): #batch内循环 但是batch为1
                target = target[target[:, 1] == 1]
                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_pred = predictions[:, :2]
                if len(target) == 0:
                    cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_target)
                    matched_indices.append([])
                    continue
                with torch.no_grad():
                    matched_row_inds, matched_col_inds = assign(predictions, target, self.img_w, self.img_h)
                    # matched_row_inds, matched_col_inds = anc_assign(predictions, target, self.img_w, self.img_h)
                    # matched_row_inds, matched_col_inds = assignOne2Many(predictions, target, self.img_w, self.img_h)
                    assert matched_row_inds.shape[0] - matched_col_inds.shape[0] == 0
                    matched_indices.append(matched_row_inds)

                    if self.last_target is not None:
                        # one-to-one match
                        matched_curr, matched_last = assignCrossFrame(target[matched_col_inds], self.last_target, self.img_w, self.img_h)
                    else:
                        matched_curr, matched_last = None, None
                    self.last_target = target[matched_col_inds].clone()

                if matched_curr is not None and len(matched_curr) != 0:
                    cls_loss_CF, reg_loss_CF, iou_loss_CF = self.crossFrameLoss(matched_row_inds, matched_col_inds, matched_curr, predictions, target)
                    cls_loss = cls_loss + cls_loss_CF
                    reg_loss = reg_loss + reg_loss_CF
                    iou_loss = iou_loss + iou_loss_CF

                # # classification targets
                # cls_target[matched_row_inds] = 1
                # cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_target) # / target.shape[0] #[240]

                # # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
                # reg_yxtl = predictions[matched_row_inds, 2:6]
                # reg_yxtl[:, 0] *= self.n_strips # start_y
                # reg_yxtl[:, 1] *= (self.img_w - 1) # start_x
                # reg_yxtl[:, 2] *= 180 # theta
                # reg_yxtl[:, 3] *= self.n_strips # length
                # target_yxtl = target[matched_col_inds, 2:6].clone()
                # target_yxtl[:, 0] *= self.n_strips
                # target_yxtl[:, 1] *= (self.img_w-1)
                # target_yxtl[:, 2] *= 180
                # target_yxtl[:, 3] *= self.n_strips
                # reg_loss = reg_loss + F.smooth_l1_loss(reg_yxtl, target_yxtl, reduction='none').mean()

                # # regression targets -> S coordinates (all transformed to absolute values)
                # reg_pred = predictions[matched_row_inds, 6:]
                # reg_pred *= (self.img_w - 1)
                # reg_targets = target[matched_col_inds, 6:].clone()
                # iou_loss = iou_loss + liou_loss(reg_pred, reg_targets, self.img_w, length=15) #15

        cls_loss /= (len(targets) * len(predictions_lists))
        reg_loss /= (len(targets) * len(predictions_lists))
        iou_loss /= (len(targets) * len(predictions_lists))
        # CrossFrameLoss /= (len(targets) * len(predictions_lists))

        return matched_indices, cls_loss, reg_loss, iou_loss#, CrossFrameLoss

    # Calculate the loss for each instance
    def CalculateInstLoss(self, matched_row, cls_loss_list, reg_loss_list, iou_loss_list):
        instLosses = []
        # for matched_row_inds, cls_loss, reg_loss, iou_loss in zip(matched_row, cls_loss_list, reg_loss_list, iou_loss_list):
        for cls_loss, reg_loss, iou_loss in zip(cls_loss_list, reg_loss_list, iou_loss_list):
            matched_row_inds = matched_row
            instLoss = 0
            instLoss += cls_loss * self.cls_weight
            if len(matched_row_inds) == 0:
                instLosses.append(instLoss)
                continue
            else:
                instLoss[matched_row_inds] += reg_loss * self.reg_weight + iou_loss * self.iou_weight   
                instLosses.append(instLoss)
        return torch.stack(instLosses, dim=0)

    def loss4OneStep(self, output, batch, diff = None):
        assert diff is not None
        targets = batch['lane_line'].clone()
        #感觉这里还有一定问题，应该用最后一列预测值来计算损失，同时如何进行match也是还没思考的问题
        matched_fir, cls_losses_A, reg_loss_A, iou_loss_A = self.line_loss_diff_A(output['predictions_fir'], targets)
        matched_sec, cls_losses_B, reg_loss_B, iou_loss_B = self.line_loss_diff_A(output['predictions_sec'], targets)

        # loss_A = self.CalculateInstLoss(matched_fir[-1], cls_losses_A, reg_losses_A, iou_losses_A) #[240]
        # loss_B = self.CalculateInstLoss(matched_sec[-1], cls_losses_B, reg_losses_B, iou_losses_B) #[240]

        diffOneFrame = torch.stack([d for d in diff], dim=0).squeeze().mean(dim=0, keepdim=False) #[240]
        # diffOneFrame = torch.stack([d for d in diff], dim=0).squeeze() # [240]
        delta_loss = torch.median(cls_losses_A - cls_losses_B).detach() #[240]
        # delta_loss = torch.median(loss_A - loss_B).detach() #[240]
        cls_losses_A = cls_losses_A - delta_loss/2
        cls_losses_B = cls_losses_B + delta_loss/2

        # total_loss = torch.sum((1 - diffOneFrame) * loss_A + diffOneFrame * loss_B)
        cls_loss = torch.sum((1 - diffOneFrame) * cls_losses_A + diffOneFrame * cls_losses_B)

        total_loss = (reg_loss_A + reg_loss_B) * self.reg_weight / 2 \
                   + (iou_loss_A + iou_loss_B) * self.iou_weight / 2 \
                   + cls_loss * self.cls_weight
                #    + CrossFrameLoss_B
        # matched_sec = self.filter_matched_list(matched_sec)
        # last_priors = torch.where((cls_losses_A>cls_losses_B).unsqueeze(0).unsqueeze(2), output['predictions_sec'][-1], output['predictions_fir'][-1])
        last_priors = output['predictions_sec'][-1][:, matched_sec[-1], :]
        return matched_sec, total_loss, last_priors

    def filter_matched_list(self, matched_indices):
        ret_matched = list()
        for matched_row_inds in matched_indices:
            mask = matched_row_inds<240
            ret_matched.append(matched_row_inds[mask])
        return ret_matched

    def crossFrameLoss(self, matched_row_inds, matched_col_inds, matched_curr, predictions, target):
        # classification targets
        cls_pred = predictions[:, :2]
        cls_target = predictions.new_zeros(predictions.shape[0]).long()
        cls_target[matched_row_inds][matched_curr] = 1
        cls_loss = self.cls_criterion(cls_pred, cls_target).sum() # / target.shape[0] #[240]

        # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
        reg_yxtl = predictions[matched_row_inds][matched_curr][..., 2:6]
        reg_yxtl[:, 0] *= self.n_strips # start_y
        reg_yxtl[:, 1] *= (self.img_w - 1) # start_x
        reg_yxtl[:, 2] *= 180 # theta
        reg_yxtl[:, 3] *= self.n_strips # length
        target_yxtl = target[matched_col_inds][matched_curr][..., 2:6].clone()
        target_yxtl[:, 0] *= self.n_strips
        target_yxtl[:, 1] *= (self.img_w-1)
        target_yxtl[:, 2] *= 180
        target_yxtl[:, 3] *= self.n_strips
        reg_loss = F.smooth_l1_loss(reg_yxtl, target_yxtl, reduction='none').mean()

        # regression targets -> S coordinates (all transformed to absolute values)
        reg_pred = predictions[matched_row_inds][matched_curr][..., 6:]
        reg_pred *= (self.img_w - 1)
        reg_targets = target[matched_col_inds][matched_curr][..., 6:].clone()
        iou_loss =  liou_loss(reg_pred, reg_targets, self.img_w, length=12) #15
        # CrossFrameLoss = cls_loss * self.cls_weight + reg_loss * self.reg_weight + iou_loss * self.iou_weight
        return cls_loss, reg_loss, iou_loss