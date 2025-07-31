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
import math
from PIL import Image
from typing import List, Dict, Optional
from uuid import uuid4

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms
from pycocotools import mask
from skimage import measure

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.lsl_backends import glip_cfg_setup, load_model, gen_captions, get_categories
from label_studio_ml.utils import get_image_local_path
from lsl_tools.sam.models.get_network import sam_model_registry
from lsl_tools.sam.conf.defaults import get_default_cfg
from lsl_tools.sam.train_net import cfg2args
from lsl_tools.sam.engine.train_sam import run_decoder
from lsl_tools.data.slide_windows import one_task_slidewindow
from lsl_tools.data.slide_windows_restore import one_task_restore, one_task_restore_semantic
from lsl_tools.cvat.lsl_inference.utils import load_mb2_model
from maskrcnn_benchmark.data.datasets.evaluation.box_aug import detect_bbox
from maskrcnn_benchmark.data.transforms import build_transforms
from segment_anything.utils.transforms import ResizeLongestSide


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sam_args_setup(args, config_file):
    cfg = get_default_cfg()
    cfg.merge_from_file(config_file)
    if args.mask_threshold != -1:
        cfg.SOLVER.THRESHOLD = args.mask_threshold
    cfg.freeze()
    args = cfg2args(args, cfg)
    return args

def find_weight(project_dir):
    suffix_to_find = "checkpoint_best.pth"
    weight_path =  f"{project_dir}/sam/Model/{suffix_to_find}"
    assert os.path.exists(weight_path), "No checkpoints found" 
    return weight_path

def gen_category(project_dir):
    coco_categories = get_categories(project_dir)
    captions = [i["name"] for i in coco_categories]
    captions = ". ".join(captions)
    categories = {idx+1:category for idx, category in enumerate(captions.split('. '))}
    return categories

def gen_sambackend(args, config_file, project_dir, labelstudio_url, token, confidence_score, sam_only):

    # Update args:
    sam_args = sam_args_setup(args, config_file)
    weight_path = find_weight(project_dir)
    sam_model = sam_model_registry["vit_b"](sam_args, checkpoint=weight_path)
    sam_model.to(device=DEVICE)
    sam_model.eval()
    # load the predictor

    # slidewindow
    slidewindow = args.slidewindow
    s_width, s_height = args.slidewindow_size
    overlap = args.overlap

    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device=DEVICE)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device=DEVICE)
    sam_img_size = 1024

    # GLIP
    if not sam_only:
        glip_config_file = f"{os.path.dirname(os.path.dirname(os.path.dirname(config_file)))}/coco/coco_glip_finetune.yaml"
        glip_cfg = glip_cfg_setup(glip_config_file, project_dir)
        glip_model = load_model(glip_cfg, project_dir)
        glip_model.eval()
        captions, positive_map_label_to_token, categories = gen_captions(glip_cfg, project_dir)
        glip_transform = build_transforms(glip_cfg, is_train=False)
    else:
        coco_categories = get_categories(project_dir, del_back=False)
        categories = {idx:category["name"] for idx, category in enumerate(coco_categories)}
        num_classes = len(coco_categories)
        mb2_model, mb2_transform = load_mb2_model(sam_args, project_dir, num_classes, device=DEVICE)
        mb2_model.eval()

    sam_transform = ResizeLongestSide(sam_img_size)
    # unset proxy
    try:
        del os.environ['http_proxy']
        del os.environ['https_proxy']
    except:
        print("No http proxy")

    class SamMLBackend(LabelStudioMLBase):
        def __init__(self, project_id, **kwargs):
            super(SamMLBackend, self).__init__(**kwargs)

            self.label = None

            self.from_name, self.to_name, self.value = None, None, None

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
            """ Returns the predicted mask for a smart keypoint that has been placed."""
            predictions = []
            self.from_name_p, self.to_name_p, self.value_p = self.get_first_tag_occurence('PolygonLabels', 'Image')
            self.from_name_r, self.to_name_r, self.value_r = self.get_first_tag_occurence('RectangleLabels', 'Image')

            # collect context information
            raw_img_path = tasks[0]['data']["image"]
            try:
                img_path = get_image_local_path(
                    raw_img_path,
                    label_studio_access_token=token,
                    label_studio_host=labelstudio_url
                )
            except:
                img_path = raw_img_path

            if not slidewindow:
                masks, labels, probs, bboxes = self.predict_base(
                    img_path=img_path,
                )
            else:
                masks, labels, probs, bboxes = self.predict_slidewindow(
                    img_path=img_path,
                )
            img = Image.open(img_path).convert('RGB')
            W, H = img.size
            predictions.append(self.get_results(
                masks=masks,
                bboxes=bboxes,
                probs=probs,
                width=W,
                height=H,
                labels=labels))

            return predictions

        def process_image(self, image):
            image = sam_transform.apply_image(image)
            image = torch.as_tensor(image, device=DEVICE)
            image = image.permute(2, 0, 1).contiguous()[None, :, :, :]
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            image = (image - pixel_mean) / pixel_std

            # Pad
            h, w = image.shape[-2:]
            padh = sam_img_size - h
            padw = sam_img_size - w
            image = F.pad(image, (0, padw, 0, padh))
            return image
        
        def binmask2polygon(self, pred_mask, width, height):
            contour = measure.find_contours(pred_mask, args.threshold)[0]
            pred_mask = []
            contour = np.flip(contour, axis=1)
            polygon = contour.ravel().reshape(-1, 2)
            x1, x2, y1, y2 = min(polygon[:, 0]), max(polygon[:, 0]), min(polygon[:, 1]), max(polygon[:, 1])
            pred_bbox = [x1, y1, x2-x1, y2-y1]
            polygon = self.smooth_polygon(polygon, pred_bbox)
            
            polygon[:, 0] = polygon[:, 0] / width * 100
            polygon[:, 1] = polygon[:, 1] / height * 100
            pred_mask = polygon.tolist()
            return pred_mask, pred_bbox
        
        def smooth_polygon(self, polygon, bbox):
            box_area = bbox[2] * bbox[3]
            num_polygon = len(polygon)
            if box_area <= 20:
                interval = 1
            else:
                interval = max(1, int(1 * num_polygon / math.sqrt(box_area)))
            return polygon[::interval,:]

        def nms(self, bboxes, scores, category_ids, masks):
            total_labels = []
            total_masks = []
            total_scores = []
            total_bboxes = []
            n_bboxes = torch.as_tensor(bboxes)
            n_scores = torch.as_tensor(scores)
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


        @torch.no_grad()
        def predict_sam(self, image, bboxes, scores, labels, mode="rle"):
            probs = []
            masks = []
            pred_bboxes = []
            category_ids = []
            height, width = image.shape[:2]
            target_size = max(width, height)
            input_image = self.process_image(image)

            """Normalize pixel values and pad to a square input."""
            imge = sam_model.image_encoder(input_image)
            bboxes = sam_transform.apply_coords(bboxes.reshape(-1, 2, 2), (width, height))
            for bbox, score, label in zip(bboxes, scores, labels):
                if score < confidence_score:
                    continue
                bbox = bbox.reshape(-1, 4)
                pred_mask = run_decoder("bbox", sam_model, imge, bbox, DEVICE, mode="Valid")
                pred_mask = F.interpolate(pred_mask, (target_size, target_size), mode="bilinear", align_corners=False)
                pred_mask = pred_mask[..., :height, :width]
                pred_mask = torch.sigmoid(pred_mask.float()).cpu()
                score = score * pred_mask.max().float().item()
                if score < confidence_score:
                    continue

                pred_mask = pred_mask > 0.9
                pred_mask = pred_mask.to(torch.uint8)[0][0].numpy()
                if pred_mask.sum().item() < 100:
                    continue

                if mode == "rle":
                    pred_mask = np.asfortranarray(pred_mask)
                    pred_mask = mask.encode(pred_mask)
                    pred_bbox = mask.toBbox(pred_mask)
                    pred_mask['counts'] = pred_mask['counts'].decode()
                    
                elif mode == "polygon":
                    pred_mask, pred_bbox = self.binmask2polygon(pred_mask, width, height)

                category_ids.append(label)
                probs.append(score)
                masks.append(pred_mask)
                pred_bboxes.append(pred_bbox)

            pred_results = self.nms(pred_bboxes, probs, category_ids, masks)
            return pred_results
        
        def gen_box_prompts(self, image: Image.Image):
            if not sam_only:
                bboxes, labels, scores = self.predict_glip(image)
            else:
                width, height = image.size
                labels = self.predict_mb2(image)
                bboxes = np.array([[0, 0, width, height]])
                scores = np.array([1.0])
            return bboxes, labels, scores
            
        @torch.no_grad()
        def predict_glip(self, image: Image.Image):
            width, height = image.size
            output = detect_bbox(
                model=glip_model,
                image=image,
                transform=glip_transform,
                device=DEVICE,
                captions=captions,
                positive_map_label_to_token=positive_map_label_to_token
            )
            output = output[0]
            output = output.resize((width, height))
            bboxes = output.bbox.cpu().numpy()
            labels = output.get_field("labels").cpu().numpy()
            scores = output.get_field("scores").cpu().numpy()
            return bboxes, labels, scores
        
        @torch.no_grad()
        def predict_mb2(self, image: Image.Image):
            image = mb2_transform(image).unsqueeze(0)
            outputs = mb2_model(image.to(DEVICE))
            _, pred_index = torch.max(outputs, dim=1)
            label = pred_index.item()
            return np.array([label])

        def predict_slidewindow(
            self,
            img_path,
            ):
            slide_results = {}
            total_masks = []
            total_labels = []
            total_scores = []
            total_bboxes = []
            img = Image.open(img_path).convert('RGB')
            W, H = img.size
            slide_imgs, img_infos = one_task_slidewindow(s_width, s_height, overlap, img_path)
            for slide_name, slide_img in slide_imgs.items():
                bboxes, labels, scores = self.gen_box_prompts(Image.fromarray(slide_img))
                slide_results[slide_name] = self.predict_sam(slide_img, bboxes, scores, labels, "rle")
            if not sam_only:
                merge_anns = one_task_restore(img_infos, slide_results, s_width, s_height, confidence_score, (W, H))
            else:
                merge_anns = one_task_restore_semantic(img_infos, slide_results, s_width, s_height, confidence_score, (W, H))

            for merge_ann in merge_anns:
                total_masks.append(merge_ann["segmentation"])
                total_labels.append(merge_ann["category_id"])
                total_scores.append(merge_ann["score"])
                total_bboxes.append(merge_ann["bbox"])
            
            return total_masks, total_labels, total_scores, total_bboxes
        
        def predict_base(
            self,
            img_path,
            ):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bboxes, labels, scores = self.gen_box_prompts(Image.fromarray(img))
            results = self.predict_sam(img, bboxes, scores, labels, "polygon")
            return results["segmentations"], results["labels"], results["scores"], results["bboxes"]

        def get_results(self, masks, bboxes, probs, width, height, labels):
            results = []
            for mask, bbox, prob, label in zip(masks, bboxes, probs, labels):
                # creates a random ID for your label everytime so no chance for errors
                label_r_id = str(uuid4())[:9]
                label_p_id = str(uuid4())[:9]
                # converting the mask from the model to RLE format which is usable in Label Studio
                results.append({
                    'id': label_r_id,
                    'from_name': self.from_name_r,
                    'to_name': self.to_name_r,
                    'original_width': width,
                    'original_height': height,
                    'image_rotation': 0,
                    'value': {
                        'rotation': 0,
                        'rectanglelabels': [categories[label]],
                        'width': bbox[2] / width * 100,
                        'height': bbox[3] / height * 100,
                        'x': bbox[0] / width * 100,
                        'y': bbox[1] / height * 100
                    },
                    'score': prob,
                    'type': 'rectanglelabels',
                    'readonly': False
                })
                results.append({
                    'id': label_p_id,
                    'from_name': self.from_name_p,
                    'to_name': self.to_name_p,
                    'original_width': width,
                    'original_height': height,
                    'image_rotation': 0,
                    'value': {
                        'points': mask,
                        'polygonlabels': [categories[label]],
                        "closed": True,
                    },
                    'score': prob,
                    'type': 'polygonlabels',
                    'readonly': False
                })

            return {
                'result': results,
            }
    return SamMLBackend
