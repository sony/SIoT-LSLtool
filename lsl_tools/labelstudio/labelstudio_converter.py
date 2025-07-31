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

import numpy as np
from skimage import measure
from tqdm import tqdm
import pycocotools.mask as mask_utils


def smooth_polygon(polygon):
    width = max(polygon[...,0]) - min(polygon[...,0])
    height = max(polygon[...,1]) - min(polygon[...,1])
    box_area = width * height
    num_polygon = len(polygon)
    if box_area <= 20:
        interval = 1
    else:
        interval = max(1, int(1 * num_polygon/math.sqrt(box_area)))
    return polygon[::interval,:]

def rle2polygon(segmentation):
    if isinstance(segmentation, list):
        polygons = segmentation
    else:
        if isinstance(segmentation["counts"], list):
        # Uncompressed RLE
            height, width = segmentation['size']
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
        bin_mask = mask_utils.decode(rle)
        contours = measure.find_contours(bin_mask, 0.5)
        polygons = []
        for contour in contours:
            contour = np.flip(contour, axis=1)
            if len(contour) <=4:
                continue
            contour = smooth_polygon(contour)
            polygon = contour.ravel().tolist()
            polygons.append(polygon)
        return polygons

def to_coco_polygon(coco_json_path, coco_polygon_json):
    with open(coco_json_path, "r") as fa:
        coco = json.load(fa)
    annotations = coco["annotations"]
    if "segmentation" in annotations[0].keys():
        for idx in tqdm(range(len(annotations))):
            segmentation = annotations[idx]["segmentation"]
            annotations[idx]["segmentation"] = rle2polygon(segmentation)

    with open(coco_polygon_json, "w+") as fb:
        json.dump(coco, fb)

def convert2labelstudio(coco_json_path, labelstudio_output):
    coco_polygon_json = f"{labelstudio_output}/preudo_label_polygon.json"
    to_coco_polygon(coco_json_path, coco_polygon_json)
    convert_script = f" label-studio-converter import coco -i {coco_polygon_json} -o {labelstudio_output}/preudo_label_labelstudio.json"
    cmd = f'{convert_script}'
    assert os.system(cmd) == 0, 'Fail to convert coco json to label studio json'
    os.remove(coco_polygon_json)