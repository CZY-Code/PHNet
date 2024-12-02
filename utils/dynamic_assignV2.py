import torch
INFINITY = 987654.0


class CLRNetIoULoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, lane_width=15 / 768):
        """
        LineIoU loss employed in CLRNet.
        Adapted from:
        https://github.com/Turoad/CLRNet/blob/main/clrnet/models/losses/lineiou_loss.py
        Args:
            weight (float): loss weight.
            lane_width (float): virtual lane half-width.
        """
        super(CLRNetIoULoss, self).__init__()
        self.loss_weight = loss_weight
        self.lane_width = lane_width

    def calc_iou(self, pred, target, pred_width, target_width):
        """
        Calculate the line iou value between predictions and targets
        Args:
            pred: lane predictions, shape: (Nl, Nr), relative coordinate
            target: ground truth, shape: (Nl, Nr), relative coordinate
            pred_width (torch.Tensor): virtual lane half-widths for prediction at pre-defined rows, shape (Nl, Nr).
            target_width (torch.Tensor): virtual lane half-widths for GT at pre-defined rows, shape (Nl, Nr).
        Returns:
            torch.Tensor: calculated IoU, shape (N).
        Nl: number of lanes, Nr: number of rows.
        """
        px1 = pred - pred_width
        px2 = pred + pred_width
        tx1 = target - target_width
        tx2 = target + target_width

        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)

        invalid_masks = (invalid_mask < 0) | (invalid_mask >= 1.0)
        ovr[invalid_masks] = 0.0
        union[invalid_masks] = 0.0
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
        return iou

    def forward(self, pred, target):
        assert (
            pred.shape == target.shape
        ), "prediction and target must have the same shape!"
        width = torch.ones_like(target) * self.lane_width
        iou = self.calc_iou(pred, target, width, width)
        return (1 - iou).mean() * self.loss_weight


class LaneIoULoss(CLRNetIoULoss):
    def __init__(self, loss_weight=1.0, lane_width=7.5 / 768, img_h=400, img_w=960):
        """
        LaneIoU loss employed in CLRerNet.
        Args:
            weight (float): loss weight.
            lane_width (float): half virtual lane width.
        """
        super(LaneIoULoss, self).__init__(loss_weight, lane_width)
        self.max_dx = 1e4
        self.img_h = img_h
        self.img_w = img_w

    def _calc_lane_width(self, pred, target):
        """
        Calculate the LaneIoU value between predictions and targets
        Args:
            pred: lane predictions, shape: (Nl, Nr), relative coordinate.
            target: ground truth, shape: (Nl, Nr), relative coordinate.
        Returns:
            torch.Tensor: virtual lane half-widths for prediction at pre-defined rows, shape (Nl, Nr).
            torch.Tensor: virtual lane half-widths for GT at pre-defined rows, shape (Nl, Nr).
        Nl: number of lanes, Nr: number of rows.
        """
        n_strips = pred.shape[1] - 1
        dy = self.img_h / n_strips * 2  # two horizontal grids
        _pred = pred.clone().detach()
        pred_dx = (_pred[:, 2:] - _pred[:, :-2]) * self.img_w  # pred x difference across two horizontal grids
        pred_width = self.lane_width * torch.sqrt(pred_dx.pow(2) + dy**2) / dy
        pred_width = torch.cat([pred_width[:, 0:1], pred_width, pred_width[:, -1:]], dim=1)
        target_dx = (target[:, 2:] - target[:, :-2]) * self.img_w
        target_dx[torch.abs(target_dx) > self.max_dx] = 0
        target_width = self.lane_width * torch.sqrt(target_dx.pow(2) + dy**2) / dy
        target_width = torch.cat([target_width[:, 0:1], target_width, target_width[:, -1:]], dim=1)

        return pred_width, target_width

    def forward(self, pred, target):
        assert (
            pred.shape == target.shape
        ), "prediction and target must have the same shape!"
        pred_width, target_width = self._calc_lane_width(pred, target)
        iou = self.calc_iou(pred, target, pred_width, target_width)
        return (1 - iou).mean() * self.loss_weight


class FocalCost:
    def __init__(self, weight=1.0, alpha=0.25, gamma=2, eps=1e-12):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
        Returns:
            torch.Tensor: cls_cost value.
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = (
            -(1 - cls_pred + self.eps).log()
            * (1 - self.alpha)
            * cls_pred.pow(self.gamma)
        )
        pos_cost = (
            -(cls_pred + self.eps).log() * self.alpha * (1 - cls_pred).pow(self.gamma)
        )
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight
    

class DistanceCost:
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, predictions, targets):
        """
        repeat predictions and targets to generate all combinations
        use the abs distance as the new distance cost
        Adapted from:
        https://github.com/Turoad/CLRNet/blob/main/clrnet/models/utils/dynamic_assign.py
        """
        num_priors = predictions.shape[0]
        num_targets = targets.shape[0]

        predictions = torch.repeat_interleave(
            predictions, num_targets, dim=0
        )  # repeat_interleave'ing [a, b] 2 times gives [a, a, b, b] ((np + nt) * 78)

        targets = torch.cat(
            num_priors * [targets]
        )  # applying this 2 times on [c, d] gives [c, d, c, d]

        invalid_masks = (targets < 0) | (targets >= 1.0)
        lengths = (~invalid_masks).sum(dim=1)
        distances = torch.abs((targets - predictions))
        distances[invalid_masks] = 0.0
        distances = distances.sum(dim=1) / (lengths.float() + 1e-9)
        distances = distances.view(num_priors, num_targets)

        return distances


class CLRNetIoUCost:
    def __init__(self, weight=1.0, lane_width=15 / 768):
        """
        LineIoU cost employed in CLRNet.
        Adapted from:
        https://github.com/Turoad/CLRNet/blob/main/clrnet/models/losses/lineiou_loss.py
        Args:
            weight (float): cost weight.
            lane_width (float): half virtual lane width.
        """
        self.weight = weight
        self.lane_width = lane_width

    def _calc_over_union(self, pred, target, pred_width, target_width):
        """
        Calculate the line iou value between predictions and targets
        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate
            target: ground truth, shape: (Nlt, Nr), relative coordinate
            pred_width (torch.Tensor): virtual lane half-widths for prediction at pre-defined rows, shape (Nl, Nr).
            target_width (torch.Tensor): virtual lane half-widths for GT at pre-defined rows, shape (Nl, Nr).
        Returns:
            torch.Tensor: calculated overlap, shape (Nlp, Nlt, Nr).
            torch.Tensor: calculated union, shape (Nlp, Nlt, Nr).
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        px1 = pred - pred_width
        px2 = pred + pred_width
        tx1 = target - target_width
        tx2 = target + target_width

        ovr = torch.min(px2[:, None, :], tx2[None, ...]) - torch.max(
            px1[:, None, :], tx1[None, ...]
        )
        union = torch.max(px2[:, None, :], tx2[None, ...]) - torch.min(
            px1[:, None, :], tx1[None, ...]
        )
        return ovr, union

    def __call__(self, pred, target):
        """
        Calculate the line iou value between predictions and targets
        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate
            target: ground truth, shape: (Nlt, Nr), relative coordinate
        Returns:
            torch.Tensor: calculated IoU matrix, shape (Nlp, Nlt)
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        ovr, union = self._calc_over_union(
            pred, target, self.lane_width, self.lane_width
        )
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        invalid_masks = (invalid_mask < 0) | (invalid_mask >= 1.0)
        ovr[invalid_masks] = 0.0
        union[invalid_masks] = 0.0
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
        return iou * self.weight


class LaneIoUCost(CLRNetIoUCost, LaneIoULoss):
    def __init__(
        self,
        weight=1.0,
        lane_width=7.5 / 768,
        img_h=384,
        img_w=768,
        use_pred_start_end=False
    ):
        """
        Angle- and length-aware LaneIoU employed in CLRerNet.
        Args:
            weight (float): cost weight.
            lane_width (float): half virtual lane width.
            use_pred_start_end (bool): apply the lane range (in horizon indices) for pred lanes
        """
        super(LaneIoUCost, self).__init__(weight, lane_width)
        self.use_pred_start_end = use_pred_start_end
        self.max_dx = 1e4
        self.img_h = img_h
        self.img_w = img_w

    @staticmethod
    def _set_invalid_with_start_end(pred, target, ovr, union, start, end, pred_width, target_width):
        """Set invalid rows for predictions and targets and modify overlaps and unions,
        with using start and end points of prediction lanes.

        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate
            target: ground truth, shape: (Nlt, Nr), relative coordinate
            ovr (torch.Tensor): calculated overlap, shape (Nlp, Nlt, Nr).
            union (torch.Tensor): calculated union, shape (Nlp, Nlt, Nr).
            start (torch.Tensor): start row indices of predictions, shape (Nlp).
            end (torch.Tensor): end row indices of predictions, shape (Nlp).
            pred_width (torch.Tensor): virtual lane half-widths for prediction at pre-defined rows, shape (Nlp, Nr).
            target_width (torch.Tensor): virtual lane half-widths for GT at pre-defined rows, shape (Nlt, Nr).

        Returns:
            torch.Tensor: calculated overlap, shape (Nlp, Nlt, Nr).
            torch.Tensor: calculated union, shape (Nlp, Nlt, Nr).
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        num_gt = target.shape[0]
        pred_mask = pred.repeat(num_gt, 1, 1).permute(1, 0, 2)
        invalid_mask_pred = (pred_mask < 0) | (pred_mask >= 1.0)
        target_mask = target.repeat(pred.shape[0], 1, 1)
        invalid_mask_gt = (target_mask < 0) | (target_mask >= 1.0)

        # set invalid-pred region using start and end
        assert start is not None and end is not None
        yind = torch.ones_like(invalid_mask_pred) * torch.arange(0, pred.shape[-1]).float().to(pred.device)
        h = pred.shape[-1] - 1
        start_idx = (start * h).long().view(-1, 1, 1) #[240, 1, 1]
        end_idx = (end * h).long().view(-1, 1, 1) #[240, 1, 1]

        invalid_mask_pred = invalid_mask_pred | (yind < start_idx) | (yind >= end_idx) #[240, 4, 72]

        # set ovr and union to zero at horizon lines where either pred or gt is missing
        invalid_mask_pred_gt = invalid_mask_pred | invalid_mask_gt
        ovr[invalid_mask_pred_gt] = 0
        union[invalid_mask_pred_gt] = 0

        # calculate virtual unions for pred-only or target-only horizon lines
        union_sep_target = target_width.repeat(pred.shape[0], 1, 1) * 2
        union_sep_pred = pred_width.repeat(num_gt, 1, 1).permute(1, 0, 2) * 2
        union[invalid_mask_pred_gt & ~invalid_mask_pred] += union_sep_pred[invalid_mask_pred_gt & ~invalid_mask_pred]
        union[invalid_mask_pred_gt & ~invalid_mask_gt] += union_sep_target[invalid_mask_pred_gt & ~invalid_mask_gt]
        return ovr, union

    @staticmethod
    def _set_invalid_without_start_end(pred, target, ovr, union):
        """Set invalid rows for predictions and targets and modify overlaps and unions,
        without using start and end points of prediction lanes.

        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate
            target: ground truth, shape: (Nlt, Nr), relative coordinate
            ovr (torch.Tensor): calculated overlap, shape (Nlp, Nlt, Nr).
            union (torch.Tensor): calculated union, shape (Nlp, Nlt, Nr).

        Returns:
            torch.Tensor: calculated overlap, shape (Nlp, Nlt, Nr).
            torch.Tensor: calculated union, shape (Nlp, Nlt, Nr).
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        target_mask = target.repeat(pred.shape[0], 1, 1)
        invalid_mask_gt = (target_mask < 0) | (target_mask >= 1.0)
        ovr[invalid_mask_gt] = 0.0
        union[invalid_mask_gt] = 0.0
        return ovr, union

    def __call__(self, pred, target, start=None, end=None):
        """
        Calculate the line iou value between predictions and targets
        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate.
            target: ground truth, shape: (Nlt, Nr), relative coordinate.
        Returns:
            torch.Tensor: calculated IoU matrix, shape (Nlp, Nlt)
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        pred_width, target_width = self._calc_lane_width(pred, target) #计算宽度
        ovr, union = self._calc_over_union(pred, target, pred_width, target_width)
        if self.use_pred_start_end is True: #iou_cost
            ovr, union = self._set_invalid_with_start_end(
                pred, target, ovr, union, start, end, pred_width, target_width
            )
        else: #iou_dynamick
            ovr, union = self._set_invalid_without_start_end(pred, target, ovr, union)
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
        return iou * self.weight


class DynamicTopkAssigner():
    """Computes dynamick-to-one lane matching between predictions and ground truth (GT).
    The dynamic k for each GT is computed using Lane(Line)IoU matrix.
    The costs matrix is calculated from:
    1) CLRNet: lane horizontal distance, starting point xy, angle and classification scores.
    2) CLRerNet: LaneIoU and classification scores.
    After the dynamick-to-one matching, the un-matched priors are treated as backgrounds.
    Thus each prior's prediction will be assigned with `0` or a positive integer
    indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_cost (dict): cls cost config
        iou_dynamick (dict): iou cost config for dynamic-k calculation
        iou_cost (dict): iou cost config
        reg_weight (float): cost weight for regression
        use_pred_length_for_iou (bool): prepare pred lane length for iou calculation.
        max_topk (int): max value for dynamic-k.
        min_topk (int): min value for dynamic-k.
    """

    def __init__(
        self,
        reg_weight=3.0,
        use_pred_length_for_iou=True,
        max_topk=4,
        min_topk=1,
    ):
        self.cls_cost = FocalCost(weight=1.0)
        self.iou_dynamick = LaneIoUCost(lane_width= 6 / 768, img_h=400, img_w=960, use_pred_start_end=False)
        self.iou_cost = LaneIoUCost(lane_width= 12 / 768, img_h=400, img_w=960, use_pred_start_end=True)
        self.use_pred_length_for_iou = use_pred_length_for_iou
        self.max_topk = max_topk
        self.min_topk = min_topk
        self.reg_weight = reg_weight

    def dynamic_k_assign(self, cost, ious_matrix):
        """
        Assign grouth truths with priors dynamically.
        Args:
            cost: the assign cost, shape (Np, Ng).
            ious_matrix: iou of grouth truth and priors, shape (Np, Ng).
        Returns:
            torch.Tensor: the indices of assigned prior.
            torch.Tensor: the corresponding ground truth indices.
        Np: number of priors (anchors), Ng: number of GT lanes.
        """
        matching_matrix = torch.zeros_like(cost)
        ious_matrix[ious_matrix < 0] = 0.0
        topk_ious, _ = torch.topk(ious_matrix, self.max_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=self.min_topk)
        num_gt = cost.shape[1]
        cost4match = cost.clone()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost4match[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[pos_idx, gt_idx] = 1.0
            cost4match[pos_idx, :] = INFINITY
        del topk_ious, dynamic_ks, pos_idx

        matched_gt = matching_matrix.sum(1)
        if (matched_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[matched_gt > 1, :], dim=1)
            matching_matrix[matched_gt > 1, :] = 0.0
            matching_matrix[matched_gt > 1, cost_argmin] = 1.0

        prior_idx = matching_matrix.sum(1).nonzero()
        gt_idx = matching_matrix[prior_idx].argmax(-1)
        return prior_idx.flatten(), gt_idx.flatten()

    def _clrernet_cost(self, predictions, targets, pred_xs, target_xs):
        """_summary_

        Args:
            predictions (Dict[torch.Trnsor]): predictions predicted by each stage, including:
                cls_logits: shape (Np, 2), anchor_params: shape (Np, 3),
                lengths: shape (Np, 1) and xs: shape (Np, Nr).
            targets (torch.Tensor): lane targets, shape: (Ng, 6+Nr).
                The first 6 elements are classification targets (2 ch), anchor starting point xy (2 ch),
                anchor theta (1ch) and anchor length (1ch).
            pred_xs (torch.Tensor): predicted x-coordinates on the predefined rows, shape (Np, Nr).
            target_xs (torch.Tensor): GT x-coordinates on the predefined rows, shape (Ng, Nr).

        Returns:
            torch.Tensor: cost matrix, shape (Np, Ng).
        Np: number of priors (anchors), Ng: number of GT lanes, Nr: number of rows.
        """
        start = end = None
        if self.use_pred_length_for_iou:
            y0 = predictions[:, 2].detach().clone()
            length = predictions[:, 5].detach().clone()
            start = y0.clamp(min=0, max=1)
            end = (start + length).clamp(min=0, max=1)
        iou_cost = self.iou_cost(pred_xs, target_xs, start, end)
        iou_score = 1 - (1 - iou_cost) / torch.max(1 - iou_cost) + 1e-2
        # classification cost
        cls_score = self.cls_cost(predictions[:, :2].detach().clone(), targets[:, 1].long())
        cost = -iou_score * self.reg_weight + cls_score
        return cost

    def assign(self, predictions, targets):
        """
        computes dynamicly matching based on the cost, including cls cost and lane similarity cost
        Args:
            predictions (Dict[torch.Trnsor]): predictions predicted by each stage, including:
                cls_logits: shape (Np, 2), anchor_params: shape (Np, 3),
                lengths: shape (Np, 1) and xs: shape (Np, Nr).
            targets (torch.Tensor): lane targets, shape: (Ng, 6+Nr).
                The first 6 elements are classification targets (2 ch), anchor starting point xy (2 ch),
                anchor theta (1ch) and anchor length (1ch).
            img_meta (dict): meta dict that includes per-image information such as image shape.
        return:
            matched_row_inds (Tensor): matched predictions, shape: (num_targets).
            matched_col_inds (Tensor): matched targets, shape: (num_targets).
        Np: number of priors (anchors), Ng: number of GT lanes, Nr: number of rows.
        """
        img_h, img_w = 384, 768

        pred_xs = predictions[:, 6:].detach().clone()
        target_xs = targets[:, 6:].detach().clone() / (img_w - 1)  # abs -> relative

        iou_dynamick = self.iou_dynamick(pred_xs, target_xs)
        cost = self._clrernet_cost(predictions, targets, pred_xs, target_xs)
        matched_row_inds, matched_col_inds = self.dynamic_k_assign(cost, iou_dynamick)

        return matched_row_inds, matched_col_inds
