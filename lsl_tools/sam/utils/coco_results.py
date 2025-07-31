# Copyright 2025 Sony Group Corporation
#
# Redistribution and use in source and binary forms, with or without modification, are permitted 
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions 
# and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
# and the following disclaimer in the documentation and/or other materials provided with the 
# distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to 
# endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np
import pycocotools.mask as mask_util
import torch
from torchvision.ops.boxes import batched_nms


def generate_coco(args, pred_masks, ind, img_id, category_ids=None, scores=None):
    anns = []
    n_bboxes = []
    n_scores = []
    n_ids = []
    n_categories = []

    for idx, pred_mask in enumerate(pred_masks):
        ann = {}
        pred_mask = torch.sigmoid(pred_mask.float())
        score = pred_mask.max().float().item()
        pred_mask = pred_mask > args.threshold
        pred_mask = pred_mask.to(torch.uint8)[0][0].numpy()
        if pred_mask.sum().item() < 100:
            continue
        if scores is not None:
            score = scores[idx] * score
        bin_mask = np.asfortranarray(pred_mask)
        bin_mask = mask_util.encode(bin_mask)
        area = mask_util.area(bin_mask)
        bbox = mask_util.toBbox(bin_mask)
        ind+=1
        ann["image_id"] = img_id
        ann["iscrowd"] = 0
        ann["id"] = ind
        ann["score"] = score
        ann["category_id"] = int(category_ids[idx]) if category_ids is not None else 1
        ann["area"] = area.tolist()
        ann["bbox"] = bbox.tolist()

        bin_mask['counts'] = bin_mask['counts'].decode()
        ann["segmentation"] = bin_mask

        anns.append(ann)
        n_bboxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
        n_categories.append(category_ids[idx] if category_ids is not None else 1)
        n_ids.append(ann["id"])
        n_scores.append(ann["score"])
    n_bboxes = torch.tensor(np.array(n_bboxes))
    n_ids = torch.tensor(np.array(n_ids))
    n_scores = torch.tensor(np.array(n_scores))
    n_categories = torch.tensor(n_categories)

    if n_bboxes.shape[0] != 0:
        keep_by_nms = batched_nms(
                    n_bboxes,
                    n_scores,
                    n_categories,
                    iou_threshold=args.iou_threshold,
                )
        n_ids = n_ids[keep_by_nms]
    nms_anns = []
    for ann in anns:
        if ann["id"] in n_ids:
            nms_anns.append(ann)

    return nms_anns, ind