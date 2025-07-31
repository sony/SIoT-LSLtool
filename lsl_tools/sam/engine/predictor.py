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
import os

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ..utils.utils import get_prompts_infos, get_class_agnost_prompts_infos
from ..utils.mean_iou import intersectionAndUnion, AverageMeter
from ..engine.train_sam import run_decoder
from ..utils.coco_results import generate_coco

img_size = 1024


def gen_mask(args, net, imge, prompts_infos, target_size, GPUdevice, origin_h, origin_w):
    preds = []
    prompts = prompts_infos["prompts"]
    num_prompts = prompts_infos["num_prompts"]
    category_ids = prompts_infos["category_ids"]
    scores = prompts_infos["scores"]
    minx, miny, maxx, maxy = prompts_infos["valid_area"]

    for idx in range(num_prompts):
        if args.mode == "bbox":
            prompt = prompts[idx][np.newaxis, ...]
            pred = run_decoder(args.mode, net, imge, prompt, GPUdevice, mode="Valid")
            pred = F.interpolate(pred, (target_size, target_size), mode="bilinear", align_corners=False)
            pred = pred[..., miny:maxy, minx:maxx]
            if args.resize_size is not None and args.padding:
                pred = F.interpolate(pred, (origin_h, origin_w), mode="bilinear", align_corners=False)
            preds.append(pred.cpu())
    return preds, category_ids, scores

def get_target_size(args, origin_w, origin_h):
    target_size = max(origin_h, origin_w)
    if args.padding and target_size < args.image_size:
        target_size = img_size
    return target_size

def predict(args, val_loader, net, imgid_dict, mode = "Validation"):
    device = torch.device('cuda:' + str(args.gpu_device))
     # eval mode
    net.eval()
    ind = 0
    anns = []
    if args.sam_only:
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
    
    if not args.sam_only or mode == "Test": # sam_only & mode == "Validation"
        assert os.path.exists(args.prompt)
        if args.mode == "bbox":
            box_prompt=COCO(args.prompt)
            image_ids = {}
            for image_info in box_prompt.dataset["images"]:
                image_ids[image_info["file_name"]] = image_info["id"]

    for batch_data in tqdm(val_loader):
        eval_start = 0

        imgsw = batch_data['image'].to(dtype = torch.float32, device = device)
        name = batch_data['image_meta_dict']['filename_or_obj'][0]
        masks = sum(batch_data['label'])
        image_id, origin_w, origin_h = imgid_dict[name]
        target_size = get_target_size(args, origin_w, origin_h)
        if isinstance(masks, int):
            continue

        if args.mode == "bbox":
            if not args.sam_only or mode == "Test":
                prompts_infos = get_prompts_infos(args, box_prompt, image_ids, name, target_size, origin_w, origin_h)
            else:
                prompts_infos = get_class_agnost_prompts_infos(args, target_size, origin_w, origin_h)
                minx, miny, maxx, maxy = prompts_infos["valid_area"]
                masks = F.interpolate(masks, (target_size, target_size), mode="bilinear", align_corners=False)
                masks = masks[..., miny:maxy, minx:maxx]
            if prompts_infos is None:
                continue
        
        evl_dist = int(imgsw.size(-1))
        imgs = imgsw[...,eval_start:eval_start + evl_dist]
        imgs = imgs.to(dtype = torch.float32, device = device)
        with torch.no_grad():
            imge = net.image_encoder(imgs)
        preds, category_ids, scores = gen_mask(args, net, imge, prompts_infos, target_size, device, origin_h, origin_w)
        sig_img_anns, ind = generate_coco(args, preds, ind, image_id, category_ids, scores)
        anns.extend(sig_img_anns)
        
        if args.sam_only:
            pred = preds[0] # only one prompt for sam_only per image
            pred[pred < args.threshold]=0
            intersection, union = intersectionAndUnion(pred.cpu(), masks.cpu(), 1)
            intersection_meter.update(intersection)
            union_meter.update(union)
    
    if not args.sam_only or mode=="Test":
        return anns
    else:
        mean_iou = intersection_meter.sum / (union_meter.sum + 1e-10)
        return anns, mean_iou[0]


