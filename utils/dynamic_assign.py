import torch
from scipy.optimize import linear_sum_assignment
INFINITY = 987654.0

def line_iou(pred, target, img_w, length=15, aligned=True):
    '''
    Calculate the line iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred, 72)
        target: ground truth, shape: (num_target, 72)
        img_w: image width
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    '''
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length

    if aligned:
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
               torch.max(px1[:, None, :], tx1[None, ...]))
        union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                 torch.min(px1[:, None, :], tx1[None, ...]))

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou

def liou_loss(pred, target, img_w, length=15):
    return (1 - line_iou(pred, target, img_w, length)).mean()

def liou_loss_diff(pred, target, img_w, length=15):
    return 1 - line_iou(pred, target, img_w, length)

def distance_cost(predictions, targets, img_w):
    """
    repeat predictions and targets to generate all combinations
    use the abs distance as the new distance cost
    """
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]
    # repeat_interleave'ing [a, b] 2 times gives [a, a, b, b] ((np + nt) * 78)
    predictions = torch.repeat_interleave(predictions, num_targets, dim=0)[..., 6:]
    # applying this 2 times on [c, d] gives [c, d, c, d]
    targets = torch.cat(num_priors * [targets])[..., 6:]

    invalid_masks = (targets < 0) | (targets >= img_w)
    lengths = (~invalid_masks).sum(dim=1)
    distances = torch.abs((targets - predictions))
    distances[invalid_masks] = 0.
    distances = distances.sum(dim=1) / (lengths.float() + 1e-9)
    distances = distances.view(num_priors, num_targets)

    return distances


def focal_cost(cls_pred, gt_labels, alpha=0.25, gamma=2, eps=1e-12):
    """
    Args:
        cls_pred (Tensor): Predicted classification logits, shape
            [num_query, num_class].
        gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

    Returns:
        torch.Tensor: cls_cost value
    """
    cls_pred = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
    pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)
    cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    return cls_cost


def dynamic_k_assign(cost, pair_wise_ious):
    """
    Assign grouth truths with priors dynamically.

    Args:
        cost: the assign cost.
        pair_wise_ious: iou of grouth truth and priors.

    Returns:
        prior_idx: the index of assigned prior.
        gt_idx: the corresponding ground truth index.
    """
    matching_matrix = torch.zeros_like(cost) #[240, num_targets]
    ious_matrix = pair_wise_ious.detach().clone()
    ious_matrix[ious_matrix < 0] = 0. #[num_priors, num_targets]
    # ious_matrix = (ious_matrix + 1.0) / 2.0
    n_candidate_k = 4
    topk_ious, topk_idx = torch.topk(ious_matrix, n_candidate_k, dim=0) #[n_candidate_k, num_targets]
    # print(topk_ious)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1) #[num_targets]

    num_gt = cost.shape[1]
    cost4match = cost.clone()
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(cost4match[:, gt_idx],
                                k=dynamic_ks[gt_idx].item(),
                                largest=False) #选择最小largest=False
        # print(pos_idx, gt_idx)
        matching_matrix[pos_idx, gt_idx] = 1.0
        cost4match[pos_idx, :] = INFINITY
    del topk_ious, dynamic_ks, pos_idx
    # print(matching_matrix)
    matched_priors = matching_matrix.sum(1) #[240]
    # print(matched_priors)
    if (matched_priors > 1).sum() > 0: #当有prior被匹配到的gt数量大于1时
        _, cost_argmin = torch.min(cost[matched_priors > 1, :], dim=1) 
        # matching_matrix[matched_gt > 1, 0] *= 0.0
        matching_matrix[matched_priors > 1, :] = 0.0
        matching_matrix[matched_priors > 1, cost_argmin] = 1.0

    prior_idx = torch.nonzero(matching_matrix.sum(1)) #寻找非零元素的位置
    gt_idx = matching_matrix[prior_idx].argmax(dim = -1)
    return prior_idx.flatten(), gt_idx.flatten()


def assign(
    predictions,
    targets,
    img_w,
    img_h,
    distance_cost_weight=3.,
    cls_cost_weight=1.,
):
    '''
    computes dynamicly matching based on the cost, including cls cost and lane similarity cost
    Args:
        predictions (Tensor): predictions predicted by each stage, shape: (num_priors, 78)
        targets (Tensor): lane targets, shape: (num_targets, 78)
    return:
        matched_row_inds (Tensor): matched predictions, shape: (num_targets)
        matched_col_inds (Tensor): matched targets, shape: (num_targets)
    '''
    predictions = predictions.detach().clone()
    # predictions[:, 3] *= (img_w - 1)
    predictions[:, 6:] *= (img_w - 1)
    targets = targets.detach().clone()

    # distances cost
    distances_score = distance_cost(predictions, targets, img_w) #[num_priors, num_targets]
    # normalize the distance
    distances_score = 1 - (distances_score / (torch.max(distances_score)+1e-4)) #72个点之间的距离

    # classification cost
    cls_score = focal_cost(predictions[:, :2], targets[:, 1].long())
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]

    target_start_xys = targets[:, 2:4]  # num_targets, 2
    target_start_xys[..., 0] *= (img_h - 1) #y
    target_start_xys[..., 1] *= (img_w - 1) #x

    prediction_start_xys = predictions[:, 2:4]
    prediction_start_xys[..., 0] *= (img_h - 1)
    prediction_start_xys[..., 1] *= (img_w - 1)

    start_xys_score = torch.cdist(prediction_start_xys, target_start_xys, p=2).reshape(num_priors, num_targets)
    start_xys_score = 1 - (start_xys_score / (torch.max(start_xys_score) + 1e-4))

    target_thetas = targets[:, 4].unsqueeze(-1)
    theta_score = torch.cdist(predictions[:, 4].unsqueeze(-1),
                              target_thetas,
                              p=1).reshape(num_priors, num_targets) * 180

    theta_score = 1 - (theta_score / (torch.max(theta_score) + 1e-4))

    cost = -(distances_score * start_xys_score * theta_score)**2 * distance_cost_weight \
           + cls_score * cls_cost_weight  #[num_priors, num_targets]

    iou = line_iou(predictions[..., 6:], targets[..., 6:], img_w, length=15, aligned=False) #[num_priors, num_targets]
    # matched_row_inds, matched_col_inds = dynamic_k_assign(cost, iou)

    ## C = (cost ** 1) * (iou ** 1) #此处的参数还要调整XXX
    C = cost - iou
    C = C.view(num_priors, num_targets).cpu()
    indices = linear_sum_assignment(C, maximize=False) #找最小
    matched_row_inds, matched_col_inds = torch.as_tensor(indices[0]), torch.as_tensor(indices[1])

    return matched_row_inds, matched_col_inds


def assignV2( #增加了invaild_len后的正样本匹配
    predictions,
    targets,
    img_w,
    img_h,
    distance_cost_weight=3.,
    cls_cost_weight=1.,
):
    predictions = predictions.detach().clone()
    predictions[:, 7:] *= (img_w - 1)
    targets = targets.detach().clone()

    # distances cost
    distances_score = distance_cost(predictions, targets, img_w) #[num_priors, num_targets]
    # normalize the distance
    distances_score = 1 - (distances_score / (torch.max(distances_score)+1e-4)) #72个点之间的距离

    # classification cost
    cls_score = focal_cost(predictions[:, :2], targets[:, 1].long())
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]

    target_start_xys = targets[:, 2:4]  # num_targets, 2
    target_start_xys[..., 0] *= (img_h - 1) #y
    target_start_xys[..., 1] *= (img_w - 1) #x

    prediction_start_xys = predictions[:, 2:4]
    prediction_start_xys[..., 0] *= (img_h - 1)
    prediction_start_xys[..., 1] *= (img_w - 1)

    start_xys_score = torch.cdist(prediction_start_xys, target_start_xys, p=2).reshape(num_priors, num_targets)
    start_xys_score = 1 - (start_xys_score / (torch.max(start_xys_score) + 1e-4))

    target_thetas = targets[:, 4].unsqueeze(-1)
    theta_score = torch.cdist(predictions[:, 4].unsqueeze(-1),
                              target_thetas,
                              p=1).reshape(num_priors, num_targets) * 180

    theta_score = 1 - (theta_score / (torch.max(theta_score) + 1e-4))

    cost = -(distances_score * start_xys_score * theta_score)**2 * distance_cost_weight \
           + cls_score * cls_cost_weight  #[num_priors, num_targets]

    iou = line_iou(predictions[..., 7:], targets[..., 7:], img_w, aligned=False) #[num_priors, num_targets]
    # matched_row_inds, matched_col_inds = dynamic_k_assign(cost, iou)

    ## C = (cost ** 1) * (iou ** 1) #此处的参数还要调整XXX
    C = cost - iou
    C = C.view(num_priors, num_targets).cpu()
    indices = linear_sum_assignment(C, maximize=False) #找最小
    matched_row_inds, matched_col_inds = torch.as_tensor(indices[0]), torch.as_tensor(indices[1])

    return matched_row_inds, matched_col_inds


def anc_assign(predictions, targets, img_w, img_h,
               distance_cost_weight=3.,
               cls_cost_weight=1.,): #2 3 4 yxt
    predictions = predictions.detach().clone()
    predictions[..., 6:] *= (img_w - 1)
    targets = targets.detach().clone()

    # distances cost
    distances_score = distance_cost(predictions, targets, img_w)
    # normalize the distance
    distances_score = 1 - (distances_score / torch.max(distances_score)) + 1e-2  

    # classification cost
    cls_score = focal_cost(predictions[:, :2], targets[:, 1].long())
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]

    target_start_xys = targets[:, 2:4]  # num_targets, 2
    target_start_xys[..., 0] *= (img_h - 1)
    target_start_xys[..., 1] *= (img_w - 1)

    prediction_start_xys = predictions[:, 2:4]
    prediction_start_xys[..., 0] *= (img_h - 1)
    prediction_start_xys[..., 1] *= (img_w - 1)

    start_xys_score = torch.cdist(prediction_start_xys, target_start_xys,
                                  p=2).reshape(num_priors, num_targets)
    start_xys_score = (1 - start_xys_score / torch.max(start_xys_score)) + 1e-2

    target_thetas = targets[:, 4].unsqueeze(-1)
    theta_score = torch.cdist(predictions[:, 4].unsqueeze(-1),
                              target_thetas,
                              p=1).reshape(num_priors, num_targets) * 180
    theta_score = (1 - theta_score / torch.max(theta_score)) + 1e-2

    cost = -(distances_score * start_xys_score * theta_score
             )**2 * distance_cost_weight + cls_score * cls_cost_weight
    
    iou = line_iou(predictions[..., 6:], targets[..., 6:], img_w, length=12, aligned=False)
    matched_row_inds, matched_col_inds = dynamic_k_assign(cost, iou)

    return matched_row_inds, matched_col_inds


def assignOne2Many(
    predictions, targets,
    img_w, img_h,
    distance_cost_weight=3.,
    cls_cost_weight=1.,):
    
    predictions = predictions.detach().clone()
    # predictions[:, 3] *= (img_w - 1)
    predictions[:, 6:] *= (img_w - 1)
    targets = targets.detach().clone()

    # distances cost
    distances_score = distance_cost(predictions, targets, img_w) #[num_priors, num_targets]
    # normalize the distance
    distances_score = 1 - (distances_score / (torch.max(distances_score) + 1e-4)) #72个点之间的距离

    # classification cost
    cls_score = focal_cost(predictions[:, :2], targets[:, 1].long(), alpha=0.5, gamma=2)
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]

    target_start_xys = targets[:, 2:4]  # num_targets, 2
    target_start_xys[..., 0] *= (img_h - 1) #y
    target_start_xys[..., 1] *= (img_w - 1) #x

    prediction_start_xys = predictions[:, 2:4]
    prediction_start_xys[..., 0] *= (img_h - 1)
    prediction_start_xys[..., 1] *= (img_w - 1)

    start_xys_score = torch.cdist(prediction_start_xys, target_start_xys, p=2).reshape(num_priors, num_targets)
    start_xys_score = 1 - (start_xys_score / (torch.max(start_xys_score) + 1e-4))

    target_thetas = targets[:, 4].unsqueeze(-1)
    theta_score = torch.cdist(predictions[:, 4].unsqueeze(-1),
                              target_thetas,
                              p=1).reshape(num_priors, num_targets) * 180

    theta_score = 1 - (theta_score / (torch.max(theta_score) + 1e-4))

    cost = -(distances_score * start_xys_score * theta_score)**2 * distance_cost_weight \
           + cls_score * cls_cost_weight  #[num_priors, num_targets]

    iou = line_iou(predictions[..., 6:], targets[..., 6:], img_w, aligned=False) #[num_priors, num_targets]
    C = cost - iou
    C = C.view(num_priors, num_targets).cpu()
    
    ious_matrix = iou.detach().clone()
    ious_matrix[ious_matrix < 0] = 0. #[num_priors, num_targets]
    # ious_matrix = (ious_matrix + 1.0) / 2.0
    n_candidate_k = 4
    topk_ious, topk_idx = torch.topk(ious_matrix, n_candidate_k, dim=0) #[n_candidate_k, num_targets]
    
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1) #[num_targets]
    dynamic_ks = dynamic_ks.cpu()
    # print(topk_ious, dynamic_ks)

    matched_row_inds = []
    matched_col_inds = []
    while(dynamic_ks.sum()>0):
        indices = linear_sum_assignment(C, maximize=False) #找最小
        row_inds, col_inds = torch.as_tensor(indices[0])[dynamic_ks>0], torch.as_tensor(indices[1])[dynamic_ks>0]
        matched_row_inds.append(row_inds)
        matched_col_inds.append(col_inds)
        dynamic_ks[dynamic_ks>0] -= 1
        C[row_inds, :] = INFINITY
    return torch.concat(matched_row_inds), torch.concat(matched_col_inds)


def assignCrossFrame(
    targets_curr,
    targets_last,
    img_w,
    img_h,
    distance_cost_weight=3.,
    cls_cost_weight=1.,
):
    
    predictions = targets_curr.detach().clone()
    targets = targets_last.detach().clone()
    # distances cost
    distances_score = distance_cost(predictions, targets, img_w)
    # normalize the distance
    distances_score = 1 - (distances_score / torch.max(distances_score)) + 1e-2  

    # classification cost
    cls_score = focal_cost(predictions[:, :2], targets[:, 1].long())
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]

    target_start_xys = targets[:, 2:4]  # num_targets, 2
    target_start_xys[..., 0] *= (img_h - 1)
    target_start_xys[..., 1] *= (img_w - 1)

    prediction_start_xys = predictions[:, 2:4]
    prediction_start_xys[..., 0] *= (img_h - 1)
    prediction_start_xys[..., 1] *= (img_w - 1)

    start_xys_score = torch.cdist(prediction_start_xys, target_start_xys,
                                  p=2).reshape(num_priors, num_targets)
    start_xys_score = (1 - start_xys_score / torch.max(start_xys_score)) + 1e-2

    target_thetas = targets[:, 4].unsqueeze(-1)
    theta_score = torch.cdist(predictions[:, 4].unsqueeze(-1),
                              target_thetas,
                              p=1).reshape(num_priors, num_targets) * 180
    theta_score = (1 - theta_score / torch.max(theta_score)) + 1e-2

    cost = -(distances_score * start_xys_score * theta_score
             )**2 * distance_cost_weight + cls_score * cls_cost_weight
    
    iou = line_iou(predictions[..., 6:], targets[..., 6:], img_w, length=12, aligned=False)
    matched_row_inds, matched_col_inds = dynamic_k_assign_CF(cost, iou)

    return matched_row_inds, matched_col_inds

def dynamic_k_assign_CF(cost, pair_wise_ious):
    """
    Assign grouth truths with priors dynamically.

    Args:
        cost: the assign cost.
        pair_wise_ious: iou of grouth truth and priors.

    Returns:
        prior_idx: the index of assigned prior.
        gt_idx: the corresponding ground truth index.
    """
    matching_matrix = torch.zeros_like(cost) #[240, num_targets]
    ious_matrix = pair_wise_ious.detach().clone()
    ious_matrix[ious_matrix < 0.8] = 0. #[num_priors, num_targets]
    ious_matrix[ious_matrix >= 0.8] = 1.
    # ious_matrix = (ious_matrix + 1.0) / 2.0
    n_candidate_k = 1
    topk_ious, topk_idx = torch.topk(ious_matrix, n_candidate_k, dim=0) #[n_candidate_k, num_targets]
    # print(topk_ious)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=0) #[num_targets]

    num_gt = cost.shape[1]
    cost4match = cost.clone()
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(cost4match[:, gt_idx],
                                k=dynamic_ks[gt_idx].item(),
                                largest=False) #选择最小largest=False
        # print(pos_idx, gt_idx)
        matching_matrix[pos_idx, gt_idx] = 1.0
        cost4match[pos_idx, :] = INFINITY
    del topk_ious, dynamic_ks, pos_idx
    # print(matching_matrix)
    matched_priors = matching_matrix.sum(1) #[240]
    # print(matched_priors)
    if (matched_priors > 1).sum() > 0: #当有prior被匹配到的gt数量大于1时
        _, cost_argmin = torch.min(cost[matched_priors > 1, :], dim=1) 
        # matching_matrix[matched_gt > 1, 0] *= 0.0
        matching_matrix[matched_priors > 1, :] = 0.0
        matching_matrix[matched_priors > 1, cost_argmin] = 1.0

    prior_idx = torch.nonzero(matching_matrix.sum(1)) #寻找非零元素的位置
    gt_idx = matching_matrix[prior_idx].argmax(dim = -1)
    return prior_idx.flatten(), gt_idx.flatten()