# ------------------------------------------------------------------------
# SeqFormer (https://github.com/wjf5203/SeqFormer)
# Copyright (c) 2021 ByteDance. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import from_numpy, nn
import torch.nn.functional as F
import torchvision.ops as ops
from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, generalized_multi_box_iou
from fvcore.nn import giou_loss

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 multi_frame: bool,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_mask: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.multi_frame = multi_frame
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_mask != 0, "all costs cant be 0"

    def forward(self, outputs, targets, nf, valid_ratios):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid() #[300, 9]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets]) #[num_insts, num_frame]
            tgt_bbox = torch.cat([v["boxes"] for v in targets]) #[num_insts, num_frame, 4]
            num_insts, num_frame = tgt_ids.shape
            tgt_bbox = tgt_bbox.reshape(num_insts, nf, 4)
            # ins_idx, frame_idx = torch.where(tgt_ids>0)
            # tgt_bbox = tgt_bbox[ins_idx, frame_idx, :] #[all_vaild_inst, 4] all_vaild_inst~[0, num_insts*num_frames]
            vaild_tgt_inst = tgt_ids.min(1)[0]
            vaild_tgt_idx = torch.where(vaild_tgt_inst<8)[0]

            target_bbox = []
            for i in range(num_insts):
                insts_bbox = []
                for j in range(num_frame):
                    if(tgt_ids[i, j]<8):
                        insts_bbox.append(tgt_bbox[i,j])
                target_bbox.append(insts_bbox)
            
            inst_av_bbox=[]
            for insts_bbox in target_bbox:
                if len(insts_bbox):
                    inst_av_bbox.append(torch.stack(insts_bbox, dim=0).sum(0)/len(insts_bbox))
            inst_av_bbox = torch.stack(inst_av_bbox, dim=0)

            out_bbox = outputs["pred_boxes"].permute(0,2,1,3).flatten(0, 1) #[1,5,300,4]->[batch_size*num_queries,nf,4]
            out_bbox = out_bbox.sum(1)

            # 拉平后多帧参数计算mutil-frame-boxes之间的距离
            # cost_bbox = torch.cdist(out_bbox.flatten(1,2), tgt_bbox.flatten(1,2)) #[300, 8]
            query_ints_dist = torch.cdist(out_bbox, inst_av_bbox) #[300, num_insts]
            cost_bbox = torch.full((num_queries, num_insts), float(10000)).to("cuda")
            cost_bbox[:, vaild_tgt_idx] = query_ints_dist

            cost_giou = 0
            tgt_bbox = torch.clip(tgt_bbox,min=1e-7,max=1)
            # for i in range(nf):
            #     cost_giou += -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[:,i]), #还没改
            #                                       box_cxcywh_to_xyxy(tgt_bbox[:,i]))
            cost_giou = cost_giou/nf

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log()) #[300, 9] 预测
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log()) #[300, 9] 预测

            # FIXME
            # cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids] #每个query对所有target的类别距离 #这里不太对
            cost_class = pos_cost_class[:, :] - neg_cost_class[:, :]

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).transpose(1,2).cpu()

            # sizes = [len(v["labels"]) for v in targets]
            # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            _, col_ind = linear_sum_assignment(C[0]) #只有一个batch
            indices = [(torch.from_numpy(col_ind)[vaild_tgt_idx].to("cuda"), vaild_tgt_idx)]

            # return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            return indices


def build_matcher(args):
    # output single frame, multi frame
    return HungarianMatcher(multi_frame=True, # True, False
                            cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)


