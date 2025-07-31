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
from typing import List, Dict, Optional
from uuid import uuid4

from PIL import Image
import torch
import numpy as np

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.data.datasets.evaluation.box_aug import detect_bbox
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.engine.inference import create_queries_and_maps_from_categories
from maskrcnn_benchmark.config import cfg
from lsl_tools.data.slide_windows import one_task_slidewindow
from lsl_tools.data.slide_windows_restore import one_task_restore

np.float = np.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

def load_model(cfg, project_dir):
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

def gen_glipbackend(args, config_file, project_dir, confidence_score, labelstudio_url, token):
    # Update the Cfg
    cfg = glip_cfg_setup(config_file, project_dir)

    # LOADING THE MODEL
    glip_model = load_model(cfg, project_dir)

    # Captions and postive_map
    captions, positive_map_label_to_token, categories = gen_captions(cfg, project_dir)

    test_transform = build_transforms(cfg, is_train=False)

    # slidewindow
    slidewindow = args.slidewindow
    s_width, s_height = args.slidewindow_size
    overlap = args.overlap

    #unset proxy
    try:
        del os.environ['http_proxy']
        del os.environ['https_proxy']
    except:
        print("No http proxy")

    class GLIPBackend(LabelStudioMLBase):

        def __init__(self, project_id, **kwargs):
            super(GLIPBackend, self).__init__(**kwargs)

            self.label = None

            self.from_name, self.to_name, self.value = None, None, None

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
            self.from_name_r, self.to_name_r, self.value_r = self.get_first_tag_occurence('RectangleLabels', 'Image')
            final_predictions = self.one_task(tasks[0])

            return final_predictions

        def predict_slidewindow(self, img_path):
            slide_results = {}
            total_points = []
            total_labels = []
            total_scores = []
            slide_imgs, img_infos = one_task_slidewindow(s_width, s_height, overlap, img_path)
            for slide_name, slide_img in slide_imgs.items():
                slide_img = Image.fromarray(slide_img)
                width, height = slide_img.size
                with torch.no_grad():
                    output = detect_bbox(
                        model=glip_model,
                        image=slide_img,
                        transform=test_transform,
                        device=DEVICE,
                        captions=captions,
                        positive_map_label_to_token=positive_map_label_to_token
                    )
                output = output[0]
                output = output.resize((width, height))
                points = output.bbox.cpu().numpy()
                labels = output.get_field("labels").cpu().numpy()
                scores = output.get_field("scores").cpu().numpy()
                slide_results[slide_name] = {"bboxes": xyxy2xywh(points), "labels": labels, "scores": scores, "segmentations": None}
            merge_anns = one_task_restore(img_infos, slide_results, s_width, s_height, confidence_score)
            for merge_ann in merge_anns:
                bbox = merge_ann["bbox"]
                bbox[2] = bbox[2] + bbox[0]
                bbox[3] = bbox[3] + bbox[1]
                total_points.append(np.array(bbox))
                total_labels.append(merge_ann["category_id"])
                total_scores.append(merge_ann["score"])
            return total_points, total_labels, total_scores

        def one_task(self, task):
            all_points = []
            all_scores = []
            all_labels = []
            all_lengths = []
            predictions = []

            raw_img_path = task['data']['image']

            try:
                img_path = get_image_local_path(
                    raw_img_path,
                    label_studio_access_token=token,
                    label_studio_host=labelstudio_url
                )
            except:
                img_path = raw_img_path

            img = Image.open(img_path).convert('RGB')
            W, H = img.size

            if slidewindow:
                points, labels, scores = self.predict_slidewindow(img_path)
            
            else:
                with torch.no_grad():
                    output = detect_bbox(
                        model=glip_model,
                        image=img,
                        transform=test_transform,
                        device=DEVICE,
                        captions=captions,
                        positive_map_label_to_token=positive_map_label_to_token
                    )
                output = output[0]
                output = output.resize((W, H))
                points = output.bbox.cpu().numpy()
                labels = output.get_field("labels").cpu().numpy()
                scores = output.get_field("scores").cpu().numpy()
                scores = [score.item() for score in scores]

            for point, label, score in zip(points, labels, scores):
                if score >= confidence_score:
                    all_points.append(point)
                    all_labels.append(label)
                    all_scores.append(score)
                    all_lengths.append((H, W))

            predictions.append(self.get_results(all_points, all_labels, all_scores, all_lengths))
            
            return predictions

        def get_results(self, all_points, all_labels, all_scores, all_lengths):
            
            results = []
            for points, labels, scores, lengths in zip(all_points, all_labels, all_scores, all_lengths):
                # random ID
                label_id = str(uuid4())[:9]

                height, width = lengths
                pred_label = int(labels)
                #TODO: add model version
                results.append({
                    'id': label_id,
                    'from_name': self.from_name_r,
                    'to_name': self.to_name_r,
                    'original_width': width,
                    'original_height': height,
                    'image_rotation': 0,
                    'value': {
                        'rotation': 0,
                        'rectanglelabels': [categories[pred_label]],
                        'width': (points[2] - points[0]) / width * 100,
                        'height': (points[3] - points[1]) / height * 100,
                        'x': points[0] / width * 100,
                        'y': points[1] / height * 100
                    },
                    'score': scores,
                    'type': 'rectanglelabels',
                    'readonly': False
                })
                

            
            return {
                'result': results
            }

    return GLIPBackend
