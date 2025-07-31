# Copyright (c) Facebook, Inc. and its affiliates.
import random

import torch
from torch.nn import functional as F
from typing import List, Optional
# from utils.losses_new import asym_unified_focal_loss


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def calculate_error_point_coords(src_masks, target_masks, select_coords):
    src_logits = point_sample(src_masks, select_coords).sigmoid()
    src_logits[src_logits>=0.9] = 1.8
    target_logits = point_sample(1.8 * target_masks, select_coords)
    error_masks = src_logits - target_logits
    # return logits
    return torch.abs(error_masks)


def get_uncertain_point_coords_with_randomness(
    coarse_logits, target_logits, gt_bbox, size_infos, uncertainty_func,num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0

    #
    # point_coords = []
    select_logits = torch.ones_like(coarse_logits) * -1000
    width = gt_bbox[3] - gt_bbox[1]
    height = gt_bbox[3] - gt_bbox[1]
    selected_area = [max(0, int(gt_bbox[0] - 0.75 * width)), max(0, int(gt_bbox[1] - 0.75 * height)),
    min(size_infos[2].item(), int(gt_bbox[2] + 0.75 * width)), min(size_infos[3].item(), int(gt_bbox[3] + 0.75 * height))]

    select_logits[:,:,selected_area[1]:selected_area[3],selected_area[0]:selected_area[2]] = coarse_logits[:,:,selected_area[1]:selected_area[3],selected_area[0]:selected_area[2]]
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio * 4)

    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    
    point_erros = calculate_error_point_coords(coarse_logits, target_logits, point_coords)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_error_points = num_points - num_uncertain_points
    # num_error_points = num_uncertain_points
    e_idx = torch.topk(point_erros[:, 0, :], k=num_error_points, dim=1)[1]
    e_shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    e_idx += e_shift[:, None]
    error_coords = point_coords.view(-1, 2)[e_idx.view(-1), :].view(num_boxes, num_error_points, 2
    )

    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )

    point_coords = cat(
    [
        point_coords,
        error_coords
    ],
    dim=1,
    )

    return point_coords

def dice_loss(
        inputs: torch.Tensor,                                                     
        targets: torch.Tensor,
        num_masks: float,
        ep=1e-8
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1) + ep
    denominator = inputs.sum(-1) + targets.sum(-1) + ep
    loss = 1 - numerator / denominator
    return loss.sum() / num_masks


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    gamma=2,
    alpha=3,
    ep=1e-8
    ):
        inputs = torch.sigmoid(inputs)
        inputs = torch.clamp(inputs, ep, 1.0 - ep)
        inputs = inputs.flatten(1)

        valid_mask = None
        # if self.ignore_index is not None:
        #     valid_mask = (targets != self.ignore_index).float()

        pos_mask = (targets == 1).float()
        neg_mask = (targets == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - inputs, gamma)).detach()
        pos_loss = -pos_weight * torch.log(inputs)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(inputs, gamma)).detach()
        neg_loss = -alpha * neg_weight * F.logsigmoid(-inputs)  # / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss / num_masks

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # return loss.mean(1).sum() / num_masks   # CE Loss from Point Select Strategy
    return loss.mean() / num_masks          # Offical CE Loss

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    # return logits
    return -(torch.abs(gt_class_logits))

def loss_masks(src_masks, target_masks, num_masks, src_bbox = None, size_infos = None, oversample_ratio=3.0):
    """Compute the losses related to the masks: the focal loss and the dice loss.
    targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    """
    # ranks = 56
    # # idx = max(0, 2 - int(128/ (((src_bbox[2] - src_bbox[0]) * (src_bbox[3] - src_bbox[1])) ** 0.5)))
    # # No need to upsample predictions as we are using normalized coordinates :)
    # padding = True
    # # if padding:
    # #     src_masks = src_masks[:,:,:256,:256]
    # #     target_masks = target_masks[:,:,:256,:256]
    # with torch.no_grad():
    #     # sample point_coords
    #     point_coords = get_uncertain_point_coords_with_randomness(
    #         src_masks,
    #         target_masks,
    #         src_bbox,
    #         size_infos,
    #         lambda logits: calculate_uncertainty(logits),
    #         ranks * ranks,
    #         oversample_ratio,
    #         0.75,
    #     )
    #     # get gt labels
    #     point_labels = point_sample(
    #         target_masks,
    #         point_coords,
    #         align_corners=False,
    #     ).squeeze(1)
    # point_logits = point_sample(
    #     src_masks,
    #     point_coords,
    #     align_corners=False,
    # ).squeeze(1)
    # loss_ce = sigmoid_ce_loss(point_logits, point_labels, num_masks)
    # loss_dice = dice_loss(point_logits, point_labels, num_masks)
    loss_dice = dice_loss(src_masks, target_masks.flatten(1), num_masks)
    loss_ce = sigmoid_ce_loss(src_masks, target_masks, num_masks)
                                                                                                                          
    del src_masks
    del target_masks
    return {"loss_dice":loss_dice, "loss_ce": loss_ce}
mse_loss = torch.nn.MSELoss()

def loss_count(src_masks, target_masks):
    return mse_loss(src_masks, target_masks)
