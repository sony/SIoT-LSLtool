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
import argparse
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from skimage import measure
from pycocotools.coco import COCO 
import pycocotools.mask as mask_utils
from tqdm import tqdm

from lsl_tools.data.slide_windows_restore import decode_segmentation


def parse_args():
    parser = argparse.ArgumentParser(description='Crop images and anns')
    parser.add_argument(
        '--img-dir', help='the origin images path')
    parser.add_argument(
        '--ann-path', default=None, help='the origin annotation json path')
    parser.add_argument(
        '--output-dir', help='the path to save new images and annotations')
    parser.add_argument(
        '--width', default=512, type=int, help='the cropped image width')
    parser.add_argument(
        '--height', default=512, type=int, help='the cropped image height')
    parser.add_argument(
        '--overlap', default=50, type=int, 
        help='the overlap of the sliding windows and should be bigger than the height and width of instances')
    parser.add_argument(
        '--percentage', default=0.2, type=float, 
        help='the minimum percentage of mask will be reserved')
    args = parser.parse_args()
    return args

# def rle2polygon(rle_segm):
#     if isinstance(rle_segm["counts"], list):
#         # Uncompressed RLE
#         height, width = rle_segm['size']
#         rle = mask_utils.frPyObjects(rle_segm, height, width)
#     else:
#         rle = rle_segm
#     bin_mask = mask_utils.decode(rle)
#     contours = measure.find_contours(bin_mask, 0.5)
#     segmentations = []
#     for contour in contours:
#         contour = np.flip(contour, axis=1)
#         segmentation = contour.ravel().tolist()
#         segmentations.append(segmentation)
#     return segmentations

def one_task_slidewindow(width, height, overlap, input, mode="path"):
    if mode == "path":
        img = cv2.imread(input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == "pil":
        img = cv2.cvtColor(np.array(input), cv2.COLOR_RGB2BGR)
    else:
        img = input
    ori_h, ori_w =  img.shape[:2]
    img_infos = {}
    slide_imgs = {}
    for idx, sx in enumerate(range(0, ori_w, width - overlap)):
        for idy, sy in enumerate(range(0, ori_h, height - overlap)):
            if sx+width<ori_w:
                lx = sx+width
            else:
                lx = ori_w
                sx = ori_w-width
            if sy+height<ori_h:
                ly = sy+height
            else:
                sy = ori_h-height
                ly = ori_h

            slide_imgs[f"{idx}_{idy}"] = img[sy:ly, sx:lx, :]
            img_infos[f"{idx}_{idy}"] = [sx, sy]
            if ly == ori_h:
                break
        if lx == ori_w:
            break
    return slide_imgs, img_infos

def slidingwindow(args, coco, img_dir, cropped_image_path):
    width = args.width
    height = args.height
    overlap = args.overlap
    if coco is not None:
        img_ids = coco.getImgIds()
    else:
        img_ids = os.listdir(img_dir)
    imgs = []
    annotations = []
    img_id = 0
    ann_id = 0
    for i in tqdm(img_ids):
        
        if coco is not None:
            info = coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            ann_ids = coco.getAnnIds(imgIds=[i])
            ann_infos = coco.loadAnns(ann_ids)
        elif i.endswith(".jpg") or i.endswith(".png"): # without annotations
            ann_infos = []
            info = {'filename': i}
        else:
            continue       

        ori_img = cv2.imread(os.path.join(img_dir, info['filename']))
        ori_h, ori_w =  ori_img.shape[:2]
        
        for idx, sx in enumerate(range(0, ori_w, width - overlap)):
            for idy, sy in enumerate(range(0, ori_h, height - overlap)):
                img = {}
                if sx+width<ori_w:
                    lx = sx+width
                else:
                    lx = ori_w
                    sx = ori_w-width
                if sy+height<ori_h:
                    ly = sy+height
                else:
                    sy = ori_h-height
                    ly = ori_h
                file_name = Path(info['filename'])
                
                img["file_name"] = f"{file_name.stem}_{idx}_{idy}{file_name.suffix}"
                img["height"] = ly-sy
                img["width"] = lx-sx
                img["id"] = img_id
                imgs.append(img)
                sliding_img = ori_img[sy:ly, sx:lx, :]
                cv2.imwrite(os.path.join(cropped_image_path, img["file_name"]), sliding_img)

                for ann in ann_infos:
                    new_ann = deepcopy(ann)
                    x, y, w, h = ann["bbox"]
                    thres = 0.5 - args.percentage
                    if sx - thres * w < (2 * x + w)/2 < lx + thres * w and sy - thres * w < (2 * y + h)/2 < ly + thres * 2:
                        max_x = min((x+w), lx)
                        max_y = min((y+h), ly)
                        min_x = max(0, x-sx)
                        min_y = max(0, y-sy)
                        if x >= lx or (x+w) <= sx or y >= ly or (y+h) <= sy:
                            continue

                        new_ann["bbox"] = [min_x, min_y, max_x-min_x-sx, max_y-min_y-sy]
                        new_ann["id"] = ann_id
                        new_ann["image_id"] = img_id
                        if "segmentation" in new_ann.keys():
                            if new_ann["segmentation"]:
                                bin_mask = decode_segmentation(new_ann["segmentation"], ori_w, ori_h)
                                bin_mask = bin_mask[sy:sy+height, sx:sx+width]
                                bin_mask = np.asfortranarray(bin_mask)
                                segm = mask_utils.encode(bin_mask)
                                segm['counts'] = segm['counts'].decode()
                                new_ann["segmentation"] = segm

                        annotations.append(new_ann)
                        ann_id+=1
                img_id+=1
                if ly == ori_h:
                    break
            if lx == ori_w:
                break
    return imgs, annotations
    

def main():
    args = parse_args()
    output_dir = args.output_dir
    img_dir = args.img_dir
    ann_path = args.ann_path
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if ann_path is None:
        ori_coco = None
    else:
        coco_label = {}
        ori_coco = COCO(ann_path)
        categories = ori_coco.dataset['categories']
    # cropped_path = f"cropped_{os.path.basename(image_path)}"
    cropped_path = f"{os.path.basename(img_dir)}"
    cropped_image_path = os.path.join(output_dir, cropped_path)
    if not os.path.exists(cropped_image_path):
        os.mkdir(cropped_image_path)
    img, annotations = slidingwindow(args, ori_coco, img_dir, cropped_image_path)
    if ann_path is not None:
        coco_label["info"] = {
            "image_width":args.width,
            "image_height":args.height,
            "patch_overlap":args.overlap,
        }
        coco_label["images"] = img
        coco_label["annotations"] = annotations
        coco_label["categories"] = categories

        if not os.path.exists(os.path.join(output_dir, "annotations")):
            os.mkdir(os.path.join(output_dir, "annotations"))
        ann_file = os.path.basename(ann_path)
        with open(os.path.join(output_dir, "annotations", ann_file), "w") as f:
            json.dump(coco_label, f)

   
if __name__ == "__main__":
    main()
