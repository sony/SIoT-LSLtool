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
from pathlib import Path
from copy import deepcopy

import cv2
import torch
import shapely
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
from collections import defaultdict
from torchvision.ops.boxes import batched_nms
from skimage import measure
from collections import Counter


def get_args():
    parser = argparse.ArgumentParser("Merge labels of cropped low-res images into high-res images")
    parser.add_argument(
        '--ann-path', default=None, help='the origin annotation json path')
    parser.add_argument(
        '--img-dir', default=None, help='the origin image root')
    parser.add_argument(
        '--split-ann-path', default=None, help='the split annotation json path')
    parser.add_argument(
        '--pred-ann-path', help='the predition json path of SAM')
    parser.add_argument(
        '--output-ann-path', help='the path to save new images and annotations')
    parser.add_argument(
        '--width', default=320, type=int, help='the cropped image width')
    parser.add_argument(
        '--height', default=320, type=int, help='the cropped image height')
    parser.add_argument(
        '--overlap', default=50, type=int, 
        help='the overlap of the sliding windows and should be bigger than the height and width of instances')
    parser.add_argument(
        '--incompleteness', default=True, type=bool, 
        help='whether to use incompleteness to filter the results')
    parser.add_argument(
        '--score-threshold', default=0.3, type=float)
    parser.add_argument(
        '--semantic', default=False, type=bool)

    return parser.parse_args()

def to_np_float64(data):
    if isinstance(data, list):
        return np.array(data, dtype=np.float64)
    else:
        return data.astype(np.float64)

def one_task_restore(img_infos, results, s_width, s_height, confidence_score, origin_size=None, mask_format="labelstudio"):
    m_anns = []
    img_id = 0
    for img_rank, result in results.items():
        start_x, start_y = img_infos[img_rank]
        bboxes = to_np_float64(result["bboxes"])
        scores = to_np_float64(result["scores"])
        labels = result["labels"]
        segmentations = result["segmentations"]
        for idx in range(len(labels)):
            img_id+=1
            score = scores[idx]
            if score < confidence_score:
                continue
            bbox = bboxes[idx]
            bbox[0] = bbox[0] + start_x
            bbox[1] = bbox[1] + start_y
            label = labels[idx]
            area = bbox[2] * bbox[3]
            segm = []
            if segmentations is not None:
                slide_mask = decode_segmentation(segmentations[idx], s_height, s_width).astype(np.uint8)
                area = slide_mask.sum()           
                if area < 100:
                    continue
                if mask_format == "cvat_mask":
                    x1 = int(bbox[0] - start_x)
                    y1 = int(bbox[1] - start_y)
                    x2 = x1 + int(bbox[2])
                    y2 = y1 + int(bbox[3])
                    segm = slide_mask[y1:y2, x1:x2].flat[:].tolist()
                    segm.extend([x1+start_x, y1+start_y ,x2-1+start_x, y2-1+start_y])
                else:
                    masks = np.zeros((origin_size[1], origin_size[0]), dtype=np.uint8)
                    masks[start_y:start_y+s_height, start_x:start_x+s_width] = slide_mask
                    if mask_format == "labelstudio":
                        polygons, _, area = mask_to_polygon(masks)
                        segm = polygons_to_labelstudio(polygons, origin_size)
                    else:
                        bin_mask = np.asfortranarray(masks)
                        segm = mask_utils.encode(bin_mask)
                        segm['counts'] = segm['counts'].decode()
                 
            m_anns.append({
                "image_id": img_rank, 
                "iscrowd": 0, 
                "id": img_id, 
                "area": area, 
                "score": score, 
                "bbox": bbox.tolist(), 
                "segmentation": segm, 
                "category_id": label
                })
    filter_anns = nms(m_anns, threshold=0.8)
    overlap_areas = get_overlap_area(s_width, s_height, img_infos)
    filter_anns = del_incomplete(filter_anns, overlap_areas=overlap_areas)
    filter_anns = nms(filter_anns, threshold=0.5)
    return filter_anns

def one_task_restore_semantic(img_infos, results, s_width, s_height, confidence_score, origin_size=None, mask_format="labelstudio"):
    m_anns = []
    areas = []
    category_ids = []
    img_id = 0
    masks = np.zeros((origin_size[1], origin_size[0]), dtype = np.uint8)
    for img_rank, result in results.items():
        start_x, start_y = img_infos[img_rank]
        img_id+=1
        scores = result["scores"]
        labels = result["labels"]
        segmentations = result["segmentations"]
        for idx in range(len(labels)):
            score = scores[idx]
            if score < confidence_score:
                continue
            if segmentations is not None:
                slide_mask = decode_segmentation(segmentations[idx], s_height, s_width).astype(np.uint8)
                if slide_mask.sum() < 100:
                    continue
                masks[start_y:start_y+s_height, start_x:start_x+s_width] += slide_mask
                category_ids.append(labels[idx])
                areas.append(slide_mask.sum())
    masks[masks>1] = 1
    area = masks.sum()
    if mask_format == "labelstudio":
        polygons, _, _ = mask_to_polygon(masks)
        segm = polygons_to_labelstudio(polygons, origin_size)
    else:
        if mask_format == "cvat_mask":
            masks = masks * 255
            segm = masks.flat[:].tolist()
            segm.extend([0, 0 ,origin_size[0]-1, origin_size[1]-1])
        else:
            bin_mask = np.asfortranarray(masks)
            segm = mask_utils.encode(bin_mask)
            segm['counts'] = segm['counts'].decode()
    
    category_id = select_category(category_ids, areas)
    m_anns.append({
        "image_id": img_rank, 
        "iscrowd": 0, 
        "id": img_id, 
        "area": area, 
        "score": score, 
        "bbox": [0, 0, origin_size[0], origin_size[1]], 
        "segmentation": segm, 
        "category_id": category_id
        })
    
    return m_anns

def mask_to_polygon(mask):
    polygons = []
    vertexes = []
    area = int((mask != 0).sum())
    contours = measure.find_contours(mask)
    for contour in contours:
        # reduce points of polygons
        poly = shapely.geometry.Polygon(contour)
        poly_s = poly.simplify(tolerance=0.1)
        contour = poly_s.boundary.coords[:]
        
        contour = np.flip(contour, axis=1)
        polygon = contour.ravel().tolist()
        polygons.append(polygon)
        x1 = np.min(contour[:,0])
        x2 = np.max(contour[:,0])
        y1 = np.min(contour[:,1])
        y2 = np.max(contour[:,1])
        vertexes.append([x1, y1])
        vertexes.append([x2, y2])
        
    # find bounding box of all polygons
    vertexes = np.array(vertexes)
    x1 = np.min(vertexes[:,0])
    x2 = np.max(vertexes[:,0])
    y1 = np.min(vertexes[:,1])
    y2 = np.max(vertexes[:,1])
    bbox = [x1, y1, x2 - x1, y2 - y1]
    return polygons, bbox, area 

def decode_segmentation(segmentation, width, height):
    if isinstance(segmentation, list):
    # Polygons
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation["counts"], list):
        # Uncompressed RLE
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return mask_utils.decode(rle).astype(np.uint8)

def polygons_to_labelstudio(polygons, origin_size):
    polygons = np.array(polygons[0])
    polygons = polygons.reshape(-1, 2)
    polygons[:, 0] = polygons[:, 0] / origin_size[0] * 100
    polygons[:, 1] = polygons[:, 1] / origin_size[1] * 100
    return polygons.tolist()

def _gen_slide_window_infos(args, img_width, img_height):
    width = args.width
    height = args.height
    overlap = args.overlap
    slide_window_infos = {}
    for idx, sx in enumerate(range(0, img_width, width - overlap)):
        for idy, sy in enumerate(range(0, img_height, height - overlap)):
            if sx+width<img_width:
                lx = sx+width
            else:
                lx = img_width
                sx = img_width-width
            if sy+height<img_height:
                ly = sy+height
            else:
                sy = img_height-height
                ly = img_height
            slide_window_infos[f"{idx}_{idy}"] = [sx, sy]
            if ly == img_height:
                break
        if lx == img_width:
            break
    return slide_window_infos

def get_overlap_area(width, height, slide_window_infos, extend=20):
    y_start = []
    y_end = []
    x_start = []
    x_end = []
    for key, value in  slide_window_infos.items():
        if key.split("_")[0] == '0':
            y_start.append(value[1])
            y_end.append(value[1] + height)
        if key.split("_")[1] == '0':
            x_start.append(value[0])
            x_end.append(value[0] + width)
    x_start = x_start[1:]
    x_end = x_end[: -1]
    y_start = y_start[1:]
    y_end = y_end[: -1]
    overlap_x_area = [[x1-extend, x2+extend] for x1, x2 in zip(x_start, x_end)]
    overlap_y_area = [[y1-extend, y2+extend] for y1, y2 in zip(y_start, y_end)]
    return [overlap_x_area, overlap_y_area]

def gen_merge_infos(pred_coco):
    merge_ids = defaultdict(list)
    merge_ranks = defaultdict(str)
    img_ids = pred_coco.getImgIds()
    img_infos = pred_coco.loadImgs(img_ids)
    
    for img_info in img_infos:
        merge_info = str(Path(img_info["file_name"]).stem).split("_")
        merge_name = "_".join(merge_info[:-2])
        merge_id = "_".join(merge_info[-2:])
        merge_ids[merge_name].append(img_info["id"])
        merge_ranks[img_info["id"]] = merge_id
    return merge_ids, merge_ranks

def get_incompleteness(pred_box, gt_box):
    lt = np.maximum(pred_box[:, None, :2], gt_box[:, :2])  # left_top (x, y)
    rb = np.minimum(pred_box[:, None, 2:], gt_box[:, 2:])  # right_bottom (x, y)
    wh = np.maximum(rb - lt + 1, 0)                # inter_area (w, h)
    inter_areas = wh[:, :, 0] * wh[:, :, 1]        # shape: (n, m)
    box_areas = (pred_box[:, 2] - pred_box[:, 0] + 1) * (pred_box[:, 3] - pred_box[:, 1] + 1)
    incompletenesses = inter_areas /  box_areas[:, None]
    row, col = np.diag_indices_from(incompletenesses)
    incompletenesses[row, col] = 0
    return incompletenesses

def xywh2xyxy(bboxes):
    bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]
    bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]
    return bboxes

def nms(anns, threshold=0.5):
    ids = []
    scores = []
    bboxes = []
    for ann in anns:
        ids.append(ann["id"])
        scores.append(ann["score"])
        bbox = ann["bbox"]
        bboxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])

    ids = torch.tensor(np.array(ids))
    scores = torch.tensor(np.array(scores), dtype=torch.float32)
    bboxes = torch.tensor(np.array(bboxes), dtype=torch.float32)

    if bboxes.shape[0] != 0:
        keep_by_nms = batched_nms(
                    bboxes,
                    scores,
                    torch.ones_like(bboxes[:, 0]),  # categories
                    iou_threshold=threshold,
                )
        ids = ids[keep_by_nms]
    nms_anns = []
    for ann in anns:
        if ann["id"] in ids:
            nms_anns.append(ann)
    return nms_anns

def filter_bboxes(anns):
    bboxes = np.array([ann["bbox"] for ann in anns])
    ann_ids =  [ann["id"] for ann in anns]
    scores = np.array([ann["score"] for ann in anns])
    if bboxes.shape[0] == 0:
        return []
    bboxes = xywh2xyxy(bboxes)
    incompletenesses = get_incompleteness(bboxes, bboxes)
    del_ann = []
    for idx, incompleteness in enumerate(incompletenesses):
        if incompleteness.max() < 0.8:
            continue
        else:
            score_intervals = scores[incompleteness >= 0.8] - scores[idx]
            iog_intervals = incompleteness[incompleteness >= 0.8] - incompletenesses[incompleteness >= 0.8][:, idx]
            for ids, score_interval in enumerate(score_intervals):
                if score_interval > -0.2 and iog_intervals[ids] > 0.1:
                    del_ann.append(ann_ids[idx])
                    break
    filter_anns = []
    for ann in anns:
        if ann["id"] not in del_ann:
            filter_anns.append(ann)
    return filter_anns

def del_incomplete(anns, overlap_areas):
    box_anns = deepcopy(anns)
    filter_ann_ids = []
    adjust_anns = []
    x_overlap_areas, y_overlap_areas = overlap_areas
    for x_area in x_overlap_areas:
        x1, x2 = x_area
        processed_anns = []
        for box_ann in box_anns:
            x, y, w, h = box_ann["bbox"]
            if x <= x1 < x+w or x< x2 <= x+w or x1 < x < x2 or x1<  x+w < x2 or x1 < x + w/2 < x2:
                processed_anns.append(box_ann)
                filter_ann_ids.append(box_ann["id"])
        box_anns = list(filter(lambda y: y["id"] not in filter_ann_ids, box_anns))
        adjust_anns.extend(filter_bboxes(processed_anns))
    
    for y_area in y_overlap_areas:
        y1, y2 = y_area
        processed_anns = []
        for box_ann in box_anns:
            x, y, w, h = box_ann["bbox"]

            if y <= y1 < y+h or y< y2 <= y+h or y1 < y < y2 or y1 < y+h < y2 or y1 < y + h/2 < y2:
                processed_anns.append(box_ann)
                filter_ann_ids.append(box_ann["id"])
        box_anns = list(filter(lambda y: y["id"] not in filter_ann_ids, box_anns))
        adjust_anns.extend(filter_bboxes(processed_anns))
    
    for ann in anns: # add bboxes without filtering
        if ann["id"] not in filter_ann_ids:
            adjust_anns.append(ann)
    return  adjust_anns

def select_category(category_ids, areas):
    collection_ids = Counter(category_ids)
    top2_count = collection_ids.most_common(2)
    category_id = top2_count[0][0]
    if len(top2_count) > 1:
        if top2_count[0][1] == top2_count[1][1]:
            total_areas = dict()
            for id, area in zip(category_ids, areas):
                if collection_ids[id] == top2_count[0][1]:
                    total_areas[id] = total_areas.setdefault(id, 0) + area
            category_id = max(total_areas, key=lambda x: total_areas[x])
    return category_id
    
      
def merge_anns(args, merge_ids, pred_coco, merge_ranks, imgname2id, name2size):
    total_anns = []
    for o_name, img_ids in tqdm(merge_ids.items()):
        m_anns = []
        img_width, img_height = name2size[o_name]
        slide_window_infos = _gen_slide_window_infos(args, img_width, img_height)
        anns = pred_coco.loadAnns(pred_coco.getAnnIds(img_ids))
        for ann in anns:
            if ann["score"] < args.score_threshold: 
                continue
            masks = np.zeros((img_height, img_width))
            segm = []
            img_id = ann["image_id"] 
            img_rank = merge_ranks[img_id]
            start_x, start_y = slide_window_infos[img_rank]
            bbox = np.array(ann["bbox"])
            bbox[::2] = bbox[::2]
            bbox[1::2] = bbox[1::2] 
            bbox[0] = bbox[0] + start_x
            bbox[1] = bbox[1] + start_y
            area = bbox[2] * bbox[3]
            if len(ann["segmentation"]) > 0:
                mask = decode_segmentation(ann["segmentation"], args.width, args.height)
                area = int(mask.sum())
                if area < 100 :
                    continue
                masks[start_y:start_y+args.height, start_x:start_x+args.width] = mask
                bin_mask = np.asfortranarray(masks.astype(np.uint8))
                segm = mask_utils.encode(bin_mask)
                segm['counts'] = segm['counts'].decode()
            
            m_anns.append({
                        "image_id": imgname2id[o_name], 
                        "iscrowd": 0, 
                        "id": ann["id"], 
                        "area": area, 
                        "score": ann["score"], 
                        "bbox": bbox.tolist(), 
                        "segmentation": segm, 
                        "category_id": ann["category_id"]
                        })
        
        filter_anns = nms(m_anns, threshold=0.8)
        if args.incompleteness:  # use incompleteness to filter the results
            overlap_areas = get_overlap_area(args.width, args.height, slide_window_infos)
            filter_anns = del_incomplete(filter_anns, overlap_areas=overlap_areas)
        total_anns.extend(nms(filter_anns, threshold=0.5))

    return total_anns

def merge_semantic_anns(args, merge_ids, pred_coco, merge_ranks, imgname2id, name2size):
    total_anns = []
    for idx, (o_name, img_ids) in enumerate(tqdm(merge_ids.items())):
        m_anns = []
        img_width, img_height = name2size[o_name]
        slide_window_infos = _gen_slide_window_infos(args, img_width, img_height)
        anns = pred_coco.loadAnns(pred_coco.getAnnIds(img_ids))
        masks = np.zeros((img_height, img_width))
        category_ids = []
        areas = []
        for ann in anns:
            img_id = ann["image_id"]
            img_rank = merge_ranks[img_id]
            start_x, start_y = slide_window_infos[img_rank]
            mask = decode_segmentation(ann["segmentation"], args.width, args.height)
            if mask.sum() < 50:
                continue
            masks[start_y:start_y+args.height, start_x:start_x+args.width] += mask
            category_ids.append(ann["category_id"])
            areas.append(mask.sum())
        masks[masks>=1] = 1
        if masks.sum() < 100:
            continue
        bin_mask = np.asfortranarray(masks.astype(np.uint8))
        segm = mask_utils.encode(bin_mask)
        segm['counts'] = segm['counts'].decode()
        category_id = select_category(category_ids, areas)
        m_anns.append({
                    "image_id": imgname2id[o_name], 
                    "iscrowd": 0, 
                    "id": idx, 
                    "area": int(masks[masks>0.5].sum()), 
                    "score": ann["score"], 
                    "bbox": [0, 0, img_width, img_height], 
                    "segmentation": segm, 
                    "category_id": category_id
                    })
        total_anns.extend(m_anns)

    return total_anns

def get_infos_from_anns(args):
    with open(args.ann_path, "r") as f:
        origin_coco = json.load(f)
    name2imgid = {str(Path(i["file_name"]).stem) : i["id"] for i in origin_coco["images"]}
    name2size = {str(Path(i["file_name"]).stem) : [i["width"], i["height"]] for i in origin_coco["images"]}
    split_coco = COCO(args.split_ann_path)
    pred_coco = split_coco.loadRes(args.pred_ann_path)
    return name2imgid, name2size, pred_coco

def get_infos_from_dir(args):
    name2imgid = {}
    name2size = {}
    result = {}
    result["images"] = []
    img_names = os.listdir(args.img_dir)
    pred_coco = COCO(args.pred_ann_path)
    result["categories"] = pred_coco.dataset["categories"]
    for idx, img_name in enumerate(img_names):
        image = cv2.imread(f"{args.img_dir}/{img_name}")
        height, width = image.shape[:2]
        file_name = str(Path(img_name).stem)
        name2imgid[file_name] = idx
        name2size[file_name] = [width, height]
        result["images"].append(
            {
                'file_name': img_name, 
                'height': height, 
                'width': width, 
                'id': idx
                }
            )
    return name2imgid, name2size, pred_coco, result

def main():
    args = get_args()
    if args.ann_path is not None:
        name2imgid, name2size, pred_coco = get_infos_from_anns(args)
    else:
        name2imgid, name2size, pred_coco, result = get_infos_from_dir(args)
        
    merge_ids, merge_ranks = gen_merge_infos(pred_coco)
    if not args.semantic:
        m_anns = merge_anns(args, merge_ids, pred_coco, merge_ranks, name2imgid, name2size)
    else:
        m_anns = merge_semantic_anns(args, merge_ids, pred_coco, merge_ranks, name2imgid, name2size)
    
    if args.ann_path is not None:
        result = m_anns
    else:
        result["annotations"] = m_anns
    with open(args.output_ann_path, "w") as fb:
        json.dump(result, fb)

if __name__ == "__main__":
    main()
