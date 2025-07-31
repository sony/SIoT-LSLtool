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
import glob
import sys
import argparse
import json
from pathlib import Path
from copy import deepcopy

import cv2
import imgviz
import numpy as np
from tqdm import tqdm
from skimage.io import imread
import torch
from PIL import Image

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.data.datasets.evaluation.box_aug import detect_bbox
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.engine.inference import create_queries_and_maps_from_categories


np.float = np.float64
SCORE_THRESHOLD = 0.002
IOU_THRESHOLD = 0.95
MAX_PROPOSAL = 100
model = None

def is_image(f):
    return Path(f).suffix.lower() in ['.jpg', '.png']    
    
def setup_lsl_glip(args, image_list_path, coco_fmt_output):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus
    cfg.merge_from_file(args.configs_file)
    if args.score_threshold is not None:
        cfg.MODEL.ATSS.INFERENCE_TH = args.score_threshold
    cfg.TEST.EVAL_TASK = "detection"
    cfg.TEST.IMS_PER_BATCH = 1
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = args.output_dir
    cfg.freeze()
    
    print(f'image path: {image_list_path}')
    if coco_fmt_output:
        print(f'coco_fmt_output: {coco_fmt_output}')
    assert not (args.weak_cap and args.caption), "Can only specify one of --weak_cap and --caption"
    return cfg

def inference_labels(args, cfg=None):
    if cfg is None:
        cfg = setup_lsl_glip(args, args.image_list_path, args.coco_fmt_output)
    
    all_captions = ['all objects', 'all entities', 'all visible entities and objects', 'all obscure entities and objects']
    if args.caption:
        all_captions = [args.caption]
    if args.weak_cap: # a:b;c:d1|d2
        weak2cap = {}
        for wc in args.weak_cap.split(';'):
            weak_name, weak_cap = wc.split(':')
            weak_caps = weak_cap.split('|')
            weak2cap[weak_name.lower()] = [x.lower() for x in weak_caps]
    

    if os.path.isfile(args.image_list_path):
        if is_image(args.image_list_path):
            images = [args.image_list_path]
        else:
            with open(args.image_list_path) as f:
                images = [x.strip() for x in f.readlines()]
    elif os.path.isdir(args.image_list_path):
        files = glob.glob(args.image_list_path + '**/*.*')
        images = []
        for f in files:
            if is_image(f):
                images.append(f)
    else:
        assert False, f"Unknown `{args.image_list_path}`"

    global model
    if model is None:
        model = build_detection_model(cfg)
        model.to(cfg.MODEL.DEVICE)
        suffix_to_find = "model_best.pth"
        weight_path =  os.path.join(cfg.OUTPUT_DIR, suffix_to_find)
        if not os.path.exists(weight_path):
            print("No checkpoints found")
            return
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
        _ = checkpointer.load(weight_path, force=True)
        model.training = False
        model.eval()
        print("Loading checkpoint successfully")

    vis_path = args.vis
    if args.vis:
        vis_path = Path(args.vis)

    categories = {idx+1:category for idx, category in enumerate(all_captions[0].split('. '))}
    captions = [i.replace("_", " ").replace("  ", " ") for i in all_captions]
    all_queries, all_positive_map_label_to_token = create_queries_and_maps_from_categories(categories, cfg)
    test_transform = build_transforms(cfg, is_train=False)
    # coco fmt 
    images_info = []
    annotations = []
    ind = 0

    categories_dir = os.path.join(cfg.OUTPUT_DIR, "cate_info.json")
    if os.path.exists(categories_dir):
        with open(categories_dir, "r") as f:
            coco_categories = json.load(f)
    else:
        coco_categories =[]
        for idx, category in enumerate(all_captions[0].split('. ')):
            coco_categories.append({"id": idx+1, "name": category, "supercategory": None})
    gt_map = {cat["name"]: cat["id"] for cat in coco_categories}
    real_map = {idx: gt_map[name] for idx, name in categories.items()}

    proposals = { 'boxes': [], 'indexes': [], 'scores':[], 'labels':[]}
    with torch.no_grad():
        for idx, fname in enumerate(pbar := tqdm(images)):
            fname = Path(fname)
            pbar.set_description_str(fname.name)
            try:
                if args.weak_cap:
                    weak_name = fname.parts[-2].lower()
                    if weak_name in weak2cap:
                        captions = weak2cap[weak_name]

                bboxes, scores = [], []
                image = Image.open(str(fname)).convert('RGB')
                output = detect_bbox(model, image, test_transform, cfg.MODEL.DEVICE, captions, all_positive_map_label_to_token[0])
                size = image.size
                output[0] = output[0].resize(size)
                bboxes = output[0].bbox.cpu().numpy()
                labels = output[0].get_field("labels").cpu().numpy()
                scores = output[0].get_field("scores").cpu().numpy()
                
                if args.coco_fmt_output:
                    images_info.append({"file_name": os.path.basename(fname), "height": image.height, "width": image.width, "id": idx})
                    coco_bboxes = deepcopy(bboxes)        
                    for ids, box in enumerate(coco_bboxes):
                        ind+=1
                        box[2] = box[2] - box[0] + 1
                        box[3] = box[3] - box[1] + 1  # XYXY to XYWH for coco fmt
                        annotation = {"id": ind, "iscrowd": 0, "image_id": idx, "bbox": box.tolist(), "category_id": real_map[int(labels[ids])], 
                        "ignore": 0, "segmentation": [], "score": float(scores[ids]), "area": float(box[2]*box[3])}
                        annotations.append(annotation)

                if vis_path:
                    if args.weak_cap:
                        img_vis_path = vis_path / '_'.join(captions)
                    else:
                        img_vis_path = vis_path
                    os.makedirs(img_vis_path, exist_ok=True)

                    try:
                        bboxes = bboxes[:, [1, 0, 3, 2]]  # XYXY to YXYX for imgviz
                        num_box = len(bboxes)
                        img = imgviz.instances2rgb(imread(fname), labels=[0,]*num_box, bboxes=bboxes, line_width=1, colormap=[[255,0,0]])
                        img_name = str(img_vis_path / fname.name)
                        cv2.imwrite(img_name, img[..., ::-1])
                        # imsave(img_name, img) # imsave extremely slow when saving 4K JPG
                    except:
                        print(f'Failed to save image to `{img_name}`')

                # proposal format
                #   proposal['boxes'] a list of Nx4 array, in XYXY format
                fname = os.path.basename(fname)
                proposals['boxes'].append(bboxes)
                proposals["indexes"].append({str(fname): idx})
                proposals['scores'].append(scores)
                proposals['labels'].append(labels)
                
                del output, image
            except Exception as e:
                print(f'Exception raised on image {fname}', file=sys.stderr)
                raise e

    if args.coco_fmt_output:
        coco_fmt_result = {"images": images_info, "annotations": annotations, "categories": coco_categories}
        with open(args.coco_fmt_output, "w") as fb:
            json.dump(coco_fmt_result, fb)
        print(f"Generate GLIP BOX Prompt in {args.coco_fmt_output}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_list_path", type=str, default="")
    ap.add_argument("--configs_file", metavar="FILE", help="path to config file",)
    ap.add_argument("--caption", required=False, type=str, default=None)
    ap.add_argument("--vis", required=False, type=str, default=None)
    ap.add_argument("--weak-cap", required=False, type=str, default=None)
    ap.add_argument("--coco_fmt_output", type=str, default=None)
    ap.add_argument("--score-threshold", type=float, default=None)
    ap.add_argument("--iou-threshold", required=False, type=float, default=IOU_THRESHOLD)
    ap.add_argument("--max-proposal", required=False, type=int, default=MAX_PROPOSAL)
    ap.add_argument("--infer-size", required=False, type=int, default=None)
    ap.add_argument("--local_rank", type=int, default=0)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    inference_labels(args, cfg=None)
