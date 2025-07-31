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
import json
import math
from typing import List, Dict, Optional

import torch
import numpy as np
from skimage import measure
from cvat_sdk import make_client
import cvat_sdk.auto_annotation as cvataa
from torchvision.ops.boxes import batched_nms
from pycocotools import mask
from torchvision import transforms

# fsod
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.engine.inference import create_queries_and_maps_from_categories
# sam
from lsl_tools.sam.conf.defaults import get_default_cfg
from lsl_tools.sam.train_net import cfg2args
from lsl_tools.sam.models.get_network import sam_model_registry as model_registry

np.float = np.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def unset_proxy():
    try:
        del os.environ['http_proxy']
        del os.environ['https_proxy']
    except:
        print("No http proxy")

#### GLIP
def glip_cfg_setup(config_file, project_dir):
    # Update the Cfg
    coco_categories = get_categories(project_dir)
    num_classes = len(coco_categories)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.num_gpus = num_gpus
    cfg.merge_from_file(config_file)
    cfg.MODEL.ATSS.INFERENCE_TH = 0.3
    cfg.TEST.EVAL_TASK = "detection"
    cfg.MODEL.ATSS.NUM_CLASSES = num_classes + 1
    cfg.MODEL.DYHEAD.NUM_CLASSES = num_classes + 1
    cfg.MODEL.FCOS.NUM_CLASSES = num_classes + 1
    cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = num_classes + 1
    cfg.TEST.IMS_PER_BATCH = 1
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.DEVICE = DEVICE
    cfg.freeze()
    return cfg

def load_glip_model(cfg, project_dir):
    # LOADING THE MODEL
    glip_model = build_detection_model(cfg)
    glip_model.to(DEVICE)
    suffix_to_find = "model_best.pth"
    weight_path =  os.path.join(project_dir, suffix_to_find)
    assert os.path.exists(weight_path), "No checkpoints found"
    checkpointer = DetectronCheckpointer(cfg, glip_model, save_dir=cfg.OUTPUT_DIR)
    _ = checkpointer.load(weight_path, force=True)
    glip_model.training = False
    glip_model.eval()
    print("Loading checkpoint successfully")
    return glip_model

def del_background(categories):
    if categories[0]["name"] == "__background__":
        categories = categories[1:]
    elif categories[0]["id"] == 0:
        for category in categories:
            category["id"]+=1
    return categories

def get_categories(project_dir, del_back=True):
    categories_dir = os.path.join(project_dir, "cate_info.json")
    assert os.path.exists(categories_dir)
    with open(categories_dir, "r") as f:
        coco_categories = json.load(f)
    if del_back:
        coco_categories = del_background(coco_categories)
    return coco_categories

def gen_captions(cfg, project_dir):
    coco_categories = get_categories(project_dir)
    captions = [i["name"] for i in coco_categories]
    captions = ". ".join(captions)
    categories = {idx+1:category for idx, category in enumerate(captions.split('. '))}
    _, all_positive_map_label_to_token = create_queries_and_maps_from_categories(categories, cfg)
    return [captions], all_positive_map_label_to_token[0], categories

def xyxy2xywh(bboxes):
    bboxes[..., 2] = bboxes[..., 2] - bboxes[..., 0]
    bboxes[..., 3] = bboxes[..., 3] - bboxes[..., 1]
    return bboxes

#### SAM
def sam_args_setup(args, config_file):
    cfg = get_default_cfg()
    cfg.merge_from_file(config_file)
    if args.mask_threshold != -1:
        cfg.SOLVER.THRESHOLD = args.mask_threshold
    cfg.freeze()
    args = cfg2args(args, cfg)
    return args

def load_mb2_model(args, project_dir, num_classes, device):
    args.num_classes = num_classes
    checkpoint = find_weight(project_dir, suffix_to_find="mb2_checkpoint_best.pth")
    mb2_model = model_registry['mobilenetv2'](args,checkpoint=checkpoint).to(device)
    mb2_transforms = transforms.Compose([ 
                transforms.Resize([args.image_size, args.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    return mb2_model, mb2_transforms

def find_weight(project_dir, suffix_to_find="checkpoint_best.pth"):
    weight_path =  f"{project_dir}/sam/Model/{suffix_to_find}"
    assert os.path.exists(weight_path), "No checkpoints found" 
    return weight_path

def binmask2polygon(pred_mask, threshold):
    bin_mask = np.asfortranarray(pred_mask)
    bin_mask = mask.encode(bin_mask)
    bbox = mask.toBbox(bin_mask).tolist()
    contours = measure.find_contours(pred_mask, threshold)
    segmentations = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        contour = smooth_polygon(contour)
        segmentation = contour.ravel().tolist()
        if len(segmentation) < 8:
            continue
        segmentations.append(segmentation)

    return segmentations, bbox

def smooth_polygon(polygon):
    width = max(polygon[...,0]) - min(polygon[...,0])
    height = max(polygon[...,0]) - min(polygon[...,0])
    box_area = width * height
    num_polygon = len(polygon)
    if box_area <= 20:
        interval = 1
    else:
        interval = max(1, int(1 * num_polygon/math.sqrt(box_area)))
    return polygon[::interval,:]

def sam_nms(bboxes, scores, category_ids, masks):
    total_labels = []
    total_masks = []
    total_scores = []
    total_bboxes = []
    n_bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
    n_scores = torch.as_tensor(scores, dtype=torch.float32)
    n_category_ids = torch.as_tensor(category_ids)
    if n_bboxes.shape[0] != 0:
        keep_by_nms = batched_nms(
                    n_bboxes,
                    n_scores,
                    n_category_ids,
                    iou_threshold=0.5,
                )
        for idx in keep_by_nms:
            total_labels.append(category_ids[idx])
            total_masks.append(masks[idx])
            total_scores.append(scores[idx])
            total_bboxes.append(bboxes[idx])
    return {"bboxes": total_bboxes, "labels": total_labels, "scores":total_scores, "segmentations":total_masks}

def get_labels(coco_json):
        labels = []
        with open(coco_json, "r") as fa:
            categories = json.load(fa)["categories"]
        for category in categories:
            category_id = category["id"]
            label_name = category["name"]
            labels.append({
                "name": label_name,
                "id": category_id,
                "attributes": []
            })
        return labels

def build_spec(coco_json, task_name):
    labels = get_labels(coco_json)
    task_spec = {
        "name": f"{task_name}",
        "labels": labels,
    }
    return task_spec