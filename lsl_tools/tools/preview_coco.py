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

from tqdm import tqdm
from pycocotools.coco import COCO
from pathlib import Path
import numpy as np
import cv2
from skimage import measure
import pycocotools.mask as maskutils


color_plate = [(0,0,0), (255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,0,255)]

def draw_bbox(image, bbox, color):
    x, y, w, h = bbox
    image = cv2.rectangle(image, (int(x), int(y), (int(w)), (int(h))) ,color, 2)
    return image

def convert_binmask(image, segms):
    height, width = image.shape[:2]
    if isinstance(segms, list):
        # Polygons
        rle = maskutils.merge(maskutils.frPyObjects(segms, height, width))
    elif isinstance(segms["counts"], list):
        # Uncompressed RLE
        rle = maskutils.frPyObjects(segms, height, width)
    else:
        rle = segms
    bin_mask = maskutils.decode(rle).astype(np.uint8)
    return bin_mask

def draw_mask(image, segms, color):
    mask = convert_binmask(image, segms)
    image[mask>0] = color
    return image

def draw_label(image, bbox, class_name, score, color):
    text_x = int(bbox[0])
    text_y = int(bbox[1])
    cv2.putText(image, f"{class_name}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
    return image

def draw_prediction(image, anns, categories, conf):
    for ann in anns:
        bbox = ann["bbox"]
        segmentation = ann["segmentation"]
        category_id = ann["category_id"]
        # class_name = categories[category_id]["name"]
        color = color_plate[category_id % len(color_plate)]
        score = ann["score"]
        if score < conf:
            continue
        if len(segmentation) > 0:
            image = draw_mask(image, segmentation, color)
        else:
            image = draw_bbox(image, bbox, color)
        # image = draw_label(image, bbox, class_name, score, color)
    return image

def visualization_coco(img_dir, annotations, output_dir, conf):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    coco = COCO(annotations)
    img_ids = coco.getImgIds()
    categories = coco.cats
    for img_id in tqdm(img_ids):
        image_name = coco.loadImgs(img_id)[0]["file_name"]
        image = cv2.imread(f"{img_dir}/{image_name}")
        ann_ids = coco.getAnnIds(img_id)
        anns = coco.loadAnns(ann_ids)
        image = draw_prediction(image, anns, categories, conf)
        cv2.imwrite(f"{output_dir}/{image_name}", image)
