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
import sys
import random
import math
import collections
import logging
import math
import os
import time
from copy import deepcopy

from PIL import Image
import torch
import numpy as np

def xywh2xyxy(bboxes):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes[..., 2] = bboxes[..., 2] + bboxes[..., 0]
    bboxes[..., 3] = bboxes[..., 3] + bboxes[..., 1]
    return bboxes

def xyxy2xywh(bboxes):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes[..., 2] = bboxes[..., 2] - bboxes[..., 0]
    bboxes[..., 3] = bboxes[..., 3] - bboxes[..., 1]
    return bboxes

def rejust_prompts(args, prompts, origin_w, origin_h):
    minx = 0
    miny = 0
    maxx = origin_w
    maxy = origin_h
    target_size = max(origin_w, origin_h)
    if args.padding and target_size < args.image_size:
        if args.resize_size is not None:
            scale = args.resize_size / max(origin_w, origin_h)
        else:
            scale = 1.0
        minx = int((args.image_size - int(origin_w * scale)) / 2)
        miny = int((args.image_size - int(origin_h * scale)) / 2)
        maxx = args.image_size - minx
        maxy = args.image_size - miny
        prompts[..., 0] = prompts[..., 0] - minx
        prompts[..., 1] = prompts[..., 1] - miny
        prompts = prompts / scale

    else:
        prompts = prompts * (target_size / args.image_size)
        if origin_h != origin_w:
            # padw = ((target_size - origin_w) * args.image_size / target_size)/2
            # pady = ((target_size - origin_h) * args.image_size / target_size)/2
            padw = (target_size - origin_w)/2
            pady = (target_size - origin_h)/2
            minx = int(padw)
            miny = int(pady)
            maxx = int(origin_w + padw)
            maxy = int(origin_h + pady)
    prompts = prompts[:,0,:]
    return prompts, minx, miny, maxx, maxy

def get_class_agnost_prompts_infos(args, target_size, origin_w, origin_h):
    prompts_infos = {}
    prompts_infos["category_ids"] = np.array([1])
    prompts_infos["scores"] = np.array([1.0])
    prompts_infos["num_prompts"] = 1
    bboxes = np.array([[0, 0, origin_w, origin_h]])
    if args.padding and target_size < args.image_size:
        bboxes, minx, miny, maxx, maxy = padding_box_prompts(args, bboxes, origin_w, origin_h)
    else:
        minx = miny = 0
        maxx = origin_w
        maxy = origin_h
        bboxes = bboxes * args.image_size / target_size
        if origin_h != origin_w:
            padw = (target_size - origin_w)/2
            pady = (target_size - origin_h)/2
            minx = int(padw)
            miny = int(pady)
            maxx = int(origin_w + padw)
            maxy = int(origin_h + pady)
            bboxes[..., 0] = bboxes[..., 0] + padw * args.image_size / target_size
            bboxes[..., 1] = bboxes[..., 1] + pady * args.image_size / target_size
            bboxes[..., 2] = bboxes[..., 2] + padw * args.image_size / target_size
            bboxes[..., 3] = bboxes[..., 3] + pady * args.image_size / target_size
    prompts_infos["prompts"] = bboxes
    prompts_infos["valid_area"] = [minx, miny, maxx, maxy]
    return prompts_infos

def get_prompts_infos(args, box_prompt, image_ids, name, target_size, origin_w, origin_h, score_threshold=0.3):
    prompts_infos = {}
    ann_ids = box_prompt.getAnnIds(image_ids[name])
    if len(ann_ids) == 0:
        return None
    ann_infos = box_prompt.loadAnns(ann_ids)
    bboxes = []
    scores = []
    category_ids = []
    for ann_info in ann_infos:
        bboxes.append(ann_info["bbox"])
        category_ids.append(ann_info["category_id"])
        if "score" not in ann_info.keys():
            scores.append(1.0)
        else:
            scores.append(ann_info["score"])
    bboxes, scores, category_ids = np.array(bboxes), np.array(scores), np.array(category_ids)
    bboxes = bboxes[scores>score_threshold]
    prompts_infos["category_ids"] = category_ids[scores>score_threshold]
    prompts_infos["scores"] = scores[scores>score_threshold]
    if bboxes.shape[0] == 0:
        return None
    bboxes = xywh2xyxy(bboxes)
    if args.padding and target_size < args.image_size:
        bboxes, minx, miny, maxx, maxy = padding_box_prompts(args, bboxes, origin_w, origin_h)
    else:
        minx = miny = 0
        maxx = origin_w
        maxy = origin_h
        bboxes = bboxes * args.image_size / target_size
        if origin_h != origin_w:
            padw = (target_size - origin_w)/2
            pady = (target_size - origin_h)/2
            minx = int(padw)
            miny = int(pady)
            maxx = int(origin_w + padw)
            maxy = int(origin_h + pady)
            bboxes[..., 0] = bboxes[..., 0] + padw * args.image_size / target_size
            bboxes[..., 1] = bboxes[..., 1] + pady * args.image_size / target_size
            bboxes[..., 2] = bboxes[..., 2] + padw * args.image_size / target_size
            bboxes[..., 3] = bboxes[..., 3] + pady * args.image_size / target_size
            
    if bboxes.shape[-1] == 0:
        return None
    prompts_infos["num_prompts"] = bboxes.shape[0]
    prompts_infos["prompts"] = bboxes
    prompts_infos["valid_area"] = [minx, miny, maxx, maxy]

    return prompts_infos

def random_click(mask, point_labels = 1, inout = 1):
    indices = np.argwhere(mask == inout)
    if len(indices) == 0:
        return None
    return indices[np.random.randint(len(indices))]


def padding_box_prompts(args, bboxes, oldw, oldh):
    if args.resize_size is not None:
        scale = args.resize_size / max(oldh, oldw)
    else:
        scale = 1.0
    padw = int((args.image_size - int(oldw * scale))/2)
    padh = int((args.image_size - int(oldh * scale))/2)
    bboxes = bboxes * scale
    bboxes[..., 0] = bboxes[..., 0] + padw
    bboxes[..., 1] = bboxes[..., 1] + padh
    bboxes[..., 2] = bboxes[..., 2] + padw
    bboxes[..., 3] = bboxes[..., 3] + padh
    minx = padw
    miny = padh
    maxx = args.image_size - minx
    maxy = args.image_size - miny

    
    return bboxes, minx, miny, maxx, maxy

def padding_valid_prompts(args,target_size, padding, bboxes, oldw, oldh):
    prompts = deepcopy(bboxes)
    prompts = np.array(prompts)
    prompts[..., 2] = prompts[..., 2] - prompts[..., 0]
    prompts[..., 3] = prompts[..., 3] - prompts[..., 1]
    minx = 0
    miny = 0
    maxx = oldw
    maxy = oldh
    if padding and target_size < args.image_size:
        if args.resize_size is not None:
            scale = args.resize_size / max(oldw, oldh)
        else:
            scale = 1.0
        minx = int((args.image_size - int(oldw * scale)) / 2)
        miny = int((args.image_size - int(oldh * scale)) / 2)
        maxx = args.image_size - minx
        maxy = args.image_size - miny
        prompts[..., 0] = prompts[..., 0] - minx
        prompts[..., 1] = prompts[..., 1] - miny
        prompts = prompts / scale

    else:
        prompts = prompts * (target_size / args.image_size)
        if oldh != oldw:
            padw = ((target_size - oldw) * args.image_size / target_size)/2
            pady = ((target_size - oldh) * args.image_size / target_size)/2
            minx = int(padw)
            miny = int(pady)
            maxx = int(oldw + padw)
            maxy = int(oldh + pady)
        prompts = prompts[:,0,:]
    return prompts, minx, miny, maxx, maxy



