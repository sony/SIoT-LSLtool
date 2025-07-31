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
from copy import deepcopy
import json
from typing import Union, Optional, List, Tuple, Text, BinaryIO, Dict, Any

import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from pycocotools.coco import COCO

from ..utils.losses import loss_masks
from ..utils.utils import rejust_prompts, xyxy2xywh
from ..utils.coco_results import generate_coco

loss_function = loss_masks
seq_backward = True

def valid_step(args, net, imgs, batch_data, masks, imgid_dict, device, pbar, ind):
    batch_loss = 0
    batch_dice_loss = 0
    batch_ce_loss = 0
    name = batch_data['image_meta_dict']['filename_or_obj'][0]
    image_id, origin_w, origin_h = imgid_dict[name]
    target_size = max(origin_w, origin_h)
    if args.padding and target_size < args.image_size:
        target_size = 1024
    preds = []

    if args.mode == "bbox":
        bboxes = batch_data['bbox'].permute(1, 0, 2)
        prompts = deepcopy(bboxes)
        prompts = xyxy2xywh(prompts)
        prompts, minx, miny, maxx, maxy = rejust_prompts(args, prompts, origin_w, origin_h)
        prompts = bboxes

    ind += 1
    '''init'''
    imgs = imgs.to(dtype = torch.float32,device = device)
    '''Train'''
    with torch.no_grad():
        imge = net.image_encoder(imgs)
        if args.mode == "bbox":
            num_bboxes = bboxes.shape[0]

        for mask, prompt in zip(masks, prompts):
            mask = mask.to(dtype = torch.float32, device = device)
            pred = run_decoder(args.mode, net, imge, prompt, device, mode="Valid")
            loss_dict = loss_function(pred, mask, num_bboxes)
            dice_loss = loss_dict["loss_dice"]
            ce_loss = loss_dict["loss_ce"]
            loss = dice_loss + ce_loss
            pred = F.interpolate(pred, (target_size, target_size), mode="bilinear", align_corners=False)
            pred = pred[..., miny:maxy, minx:maxx]
            if args.resize_size is not None and args.padding:
                pred = F.interpolate(pred, (origin_h, origin_w), mode="bilinear", align_corners=False)
            preds.append(pred.cpu())

            pbar.set_postfix(**{'Valid loss (batch)': loss.item(), 'pred.max': pred.max().item()})
            batch_loss += loss.item()
            batch_dice_loss += dice_loss.item()
            batch_ce_loss += ce_loss.item()  
    del imge
    return preds, batch_loss, batch_dice_loss, batch_ce_loss, ind

def train_step(args, net, imgs, masks, prompts, device, scaler, optimizer, pbar, ind):
    batch_loss = 0
    batch_dice_loss = 0
    batch_ce_loss = 0
    num_prompts = prompts.shape[0]
    with autocast():
        imge = net.image_encoder(imgs)

    if seq_backward:
        mask_features = imge.detach()
        mask_features.requires_grad = True
        mask_features.retain_grad()
        imge = mask_features
        
    for mask, prompt in zip(masks, prompts):
        mask = mask.to(dtype = torch.float32, device = device)
        with autocast():
            pred = run_decoder(args.mode, net, imge, prompt, device)
            loss_dict = loss_function(pred, mask, num_prompts)
            dice_loss = loss_dict["loss_dice"]
            ce_loss = loss_dict["loss_ce"]
            loss = (dice_loss + ce_loss)

        scaler.scale(loss).backward(retain_graph=True)
        ind += 1
        if not seq_backward:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        pbar.set_postfix(**{'loss (batch)': loss.item(), 'pred.max': pred.max().item()})
        batch_loss += loss.item()
        batch_dice_loss += dice_loss.item()
        batch_ce_loss += ce_loss.item()  
            
    if seq_backward:
        torch.autograd.backward(tensors=[imge],
                            grad_tensors=[imge.grad])
        del imge
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    pbar.update()
    return batch_loss, batch_dice_loss, batch_ce_loss, ind

def run_one_batch(args, net, batch_data, pbar, optimizer, scaler, device, ind, imgid_dict=None, mode="Train"):
    
    imgs = batch_data['image'].to(dtype = torch.float32, device = device)
    masks = batch_data['label']
    
    ind += 1
    '''init'''
    mask_type = torch.float32
    imgs = imgs.to(dtype = mask_type,device = device)
    
    if mode == "Train":
        if args.mode == "bbox":
            if batch_data["bbox"].shape[1] == 0:
                return False, False, False, False
            bboxes = batch_data['bbox'].permute(1, 0, 2)
            prompts = bboxes
        return train_step(args, net, imgs, masks, prompts, device, scaler, optimizer, pbar, ind)
    else:
        
        return valid_step(args, net, imgs, batch_data, masks, imgid_dict, device, pbar, ind)

def run_decoder(prompt_mode, net, imge, prompt, GPUdevice, mode="Training", target_size=1024):
    if prompt_mode == "bbox":
        pt = None
        bbox = prompt
        if bbox.sum().item() == 0:
            box_torch = None
        else:
            box_torch = torch.as_tensor(bbox, dtype=torch.float, device=GPUdevice)
    with torch.no_grad():
        se, de = net.prompt_encoder(
            points=pt,
            boxes=box_torch,
            masks=None,
        )
    if mode == "Training":
        grad_enabled = True
    else:
        grad_enabled = False
    with torch.set_grad_enabled(grad_enabled):
        pred, iou = net.mask_decoder(
        image_embeddings=imge,
        image_pe=net.prompt_encoder.get_dense_pe(), 
        sparse_prompt_embeddings=se,
        dense_prompt_embeddings=de, 
        multimask_output=False,
        )

    pred = F.interpolate(pred, (target_size, target_size), mode="bilinear", align_corners=False)    
    return pred

def train_sam(args, net, optimizer, scheduler, train_loader, epoch, scaler):
    device = torch.device('cuda:' + str(args.gpu_device))
    
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()
    if args.freeze:
        for n, value in net.image_encoder.named_parameters():
            if "Adapter" not in n and "adaptmlp" not in n:
                value.requires_grad = False
    epoch_loss = 0
    epoch_dice_loss = 0
    epoch_ce_loss = 0

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for batch_data in train_loader:
            masks = batch_data["label"]
            if len(masks) == 0:
                continue
            batch_loss, batch_dice_loss, batch_ce_loss, ind = run_one_batch(args, net, batch_data, pbar, optimizer, scaler, device, ind)
            epoch_loss+=batch_loss
            epoch_dice_loss+=batch_dice_loss
            epoch_ce_loss+=batch_ce_loss
    scheduler.step()
    return epoch_loss/ind, epoch_dice_loss/ind, epoch_ce_loss/ind

def valid_sam(args, net, val_loader, imgid_dict):
    device = torch.device('cuda:' + str(args.gpu_device))
    ind = 0
    net.eval()    
    epoch_loss = 0
    epoch_dice_loss = 0
    epoch_ce_loss = 0
    anns = []

    with tqdm(total=len(val_loader), desc='Validation round', unit='batch', leave=False) as pbar:
        for batch_data in val_loader:
            name = batch_data['image_meta_dict']['filename_or_obj'][0]
            category_ids = batch_data['category_ids'].tolist()[0] if "category_ids" in batch_data.keys() else None
            masks = batch_data["label"]
            if len(masks) == 0:
                continue
            image_id, _, _ = imgid_dict[name]
            preds, batch_loss, batch_dice_loss, batch_ce_loss, ind = run_one_batch(args, net, batch_data, pbar, None, None, device, ind, imgid_dict, mode="Valid")
            epoch_loss+=batch_loss
            epoch_dice_loss+=batch_dice_loss
            epoch_ce_loss+=batch_ce_loss
            sig_img_anns, ind = generate_coco(args, preds, ind, image_id, category_ids)
            anns.extend(sig_img_anns)
            pbar.update()
    
    return anns, epoch_loss/ind, epoch_dice_loss/ind, epoch_ce_loss/ind