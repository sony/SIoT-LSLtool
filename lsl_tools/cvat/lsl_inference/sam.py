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
from cvat_sdk import make_client
import cvat_sdk.models as models
import cvat_sdk.auto_annotation as cvataa
from cvat_sdk.datasets.task_dataset import TaskDataset

# glip
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.data.datasets.evaluation.box_aug import detect_bbox
from lsl_tools.cvat.lsl_inference.utils import load_glip_model, glip_cfg_setup, gen_captions, get_categories
# sam
from lsl_tools.data.slide_windows import one_task_slidewindow
from lsl_tools.data.slide_windows_restore import one_task_restore, one_task_restore_semantic
from lsl_tools.sam.engine.train_sam import run_decoder
from lsl_tools.sam.models.get_network import sam_model_registry
from lsl_tools.cvat.lsl_inference.utils import sam_args_setup, find_weight, sam_nms, unset_proxy, load_mb2_model
from segment_anything.utils.transforms import ResizeLongestSide


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device=DEVICE)
pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device=DEVICE)
sam_img_size = 1024


class SamSegmentationFunction:
    def __init__(self, args, config_file: str, project_dir: str, confidence_score: float, slidewindow: bool, slidewindow_size: tuple, overlap: int, sam_only: bool, cvat_mask:bool) -> None:

        #sam
        sam_args = sam_args_setup(args, config_file)
        sam_weight_path = find_weight(project_dir)
        self.sam_model = sam_model_registry["vit_b"](sam_args, checkpoint=sam_weight_path)
        self.sam_model.to(device=DEVICE)
        self.sam_model.eval()
        self.sam_transform = ResizeLongestSide(sam_img_size)
        # glip
        self.sam_only = sam_only
        self.cvat_mask = cvat_mask
        if not sam_only:
            self.categories = get_categories(project_dir)
            glip_config_file = f"{os.path.dirname(os.path.dirname(os.path.dirname(config_file)))}/coco/coco_glip_finetune.yaml"
            glip_cfg = glip_cfg_setup(glip_config_file, project_dir)
            self.glip_model = load_glip_model(glip_cfg, project_dir)
            self.glip_model.eval()
            self.captions, self.positive_map_label_to_token, _ = gen_captions(glip_cfg, project_dir)
            self.glip_transform = build_transforms(glip_cfg, is_train=False)
        else:
            self.categories = get_categories(project_dir, del_back=False)
            num_classes = len(self.categories)
            self.mb2_model, self.mb2_transform = load_mb2_model(sam_args, project_dir, num_classes, device=DEVICE)
            self.mb2_model.eval()
            self.index2category = {idx:category["id"] for idx, category in enumerate(self.categories)}

        # slidewindow
        self.slidewindow = slidewindow
        self.s_width, self.s_height = slidewindow_size
        self.overlap = overlap
        
        self.sam_threshold = args.threshold
        self.confidence_score = confidence_score
        
        unset_proxy

    def predict(self, image: Image.Image):
        
        if not self.slidewindow:
            bboxes, labels, scores = self.gen_box_prompts(image)
            # PIL to cv2
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # run the ML model
            results = self.predict_sam(image, bboxes, scores, labels, cvat_mask=self.cvat_mask)
        else:
            results = self.predict_slidewindow(image)
        return results

    def inference(self, dataset):
        images = []
        annotations = []
        ind = 0
        for sample in dataset.samples:
            image_id = sample.frame_index
            file_name = sample.frame_name
            image = dataset._load_frame_image(image_id)
            width, height = image.size
            results = self.predict(image)
            anns, ind = self.convert_coco_fmt(results, image_id, ind)
            annotations.extend(anns)
            images.append({
                "file_name": file_name, 
                "height": height, 
                "width": width, 
                "id": image_id})
        return {"images": images, "annotations": annotations, "categories": self.categories}
            
    def convert_coco_fmt(self, results, image_id, ind):
        anns = []
        for bbox, label, score, segm in zip(results["bboxes"], results["labels"], results["scores"], results["segmentations"]):
            ind+=1
            anns.append({
                "image_id": image_id, 
                "iscrowd": 0, 
                "id": ind, 
                "area": bbox[2] * bbox[3], 
                "score": score, 
                "bbox": bbox, 
                "segmentation": segm, 
                "category_id": int(label)
                })
        return anns, ind

    def inference_cvat(self, task, dataset, project_dir):
        coco_results = self.inference(dataset)
        tmp_coco_path = self.save_tmp_results(project_dir, coco_results)
        # task_spec = build_spec(coco_results, "lsl-sam")
        task.import_annotations("COCO 1.0", tmp_coco_path, pbar=None)
        task.fetch()
        os.remove(tmp_coco_path)

    def save_tmp_results(self, project_dir, coco_results):
        tmp_coco_path = f"{project_dir}/cvat_tmp.json"
        with open(tmp_coco_path, "w") as f:
            json.dump(coco_results, f)
        return tmp_coco_path

    def process_image(self, image):
        image = self.sam_transform.apply_image(image)
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
    
    def gen_box_prompts(self, image: Image.Image):
        if not self.sam_only:
            bboxes, labels, scores = self.predict_glip(image)
        else:
            width, height = image.size
            labels = self.predict_mb2(image)
            bboxes = np.array([[0.0, 0.0, width, height]]).astype(np.float64)
            scores = np.array([1.0])
        return bboxes, labels, scores

    @torch.no_grad()
    def predict_mb2(self, image: Image.Image):
        image = self.mb2_transform(image).unsqueeze(0)
        outputs = self.mb2_model(image.to(DEVICE))
        _, pred_index = torch.max(outputs, dim=1)
        label = self.index2category[pred_index.item()]
        return np.array([label])
        
    @torch.no_grad()
    def predict_glip(self, image: Image.Image):
        width, height = image.size
        results = detect_bbox(
            model=self.glip_model,
            image=image,
            transform=self.glip_transform,
            device=DEVICE,
            captions=self.captions,
            positive_map_label_to_token=self.positive_map_label_to_token
        )
        results = results[0].resize((width, height))
        bboxes = results.bbox.cpu().numpy()
        labels = results.get_field("labels").cpu().numpy().tolist()
        scores = results.get_field("scores").cpu().numpy().tolist()
        return bboxes, labels, scores

    @torch.no_grad()
    def predict_sam(self, image, bboxes, scores, labels, min_size=100, cvat_mask=False):
        probs = []
        masks = []
        pred_bboxes = []
        category_ids = []
        height, width = image.shape[:2]
        target_size = max(width, height)
        input_image = self.process_image(image)

        """Normalize pixel values and pad to a square input."""
        imge = self.sam_model.image_encoder(input_image)
        bboxes = self.sam_transform.apply_coords(bboxes.reshape(-1, 2, 2), (width, height))
        for bbox, score, label in zip(bboxes, scores, labels):
            if score < self.confidence_score:
                continue
            bbox = bbox.reshape(-1, 4)
            pred_mask = run_decoder("bbox", self.sam_model, imge, bbox, DEVICE, mode="Valid")
            pred_mask = F.interpolate(pred_mask, (target_size, target_size), mode="bilinear", align_corners=False)
            pred_mask = pred_mask[..., :height, :width]
            pred_mask = torch.sigmoid(pred_mask.float()).cpu()
            score = score * (pred_mask.max().float()).item()
            if score < self.confidence_score:
                continue
            
            pred_mask = pred_mask > self.sam_threshold
            pred_mask = pred_mask.to(torch.uint8)[0][0].numpy()
            if pred_mask.sum().item() < min_size:
                continue

            bin_mask = np.asfortranarray(pred_mask)
            pred_mask = mask.encode(bin_mask)
            pred_bbox = mask.toBbox(pred_mask).astype(np.float32)
            
            if cvat_mask:
                # cvat format mask
                cvat_bbox = pred_bbox
                if self.sam_only:
                    cvat_bbox = [0, 0, width-1, height-1]
                    pred_mask = bin_mask.flat[:].tolist()
                else:
                    x1 = int(cvat_bbox[0])
                    y1 = int(cvat_bbox[1])
                    x2 = int(cvat_bbox[0] + cvat_bbox[2])
                    y2 = int(cvat_bbox[1] + cvat_bbox[3])
                    pred_mask = bin_mask[y1:y2, x1:x2].flat[:].tolist()
                    cvat_bbox = [x1, y1, x2-1, y2-1]
                pred_mask.extend(cvat_bbox)
            else:
                # coco rle format mask 
                pred_mask['counts'] = pred_mask['counts'].decode()

            category_ids.append(label)
            probs.append(score)
            masks.append(pred_mask)
            pred_bboxes.append(pred_bbox.tolist())

        pred_results = sam_nms(pred_bboxes, probs, category_ids, masks)
        return pred_results

    def predict_slidewindow(
        self,
        image,
        ):
        slide_results = {}
        total_masks = []
        total_labels = []
        total_scores = []
        total_bboxes = []
        W, H = image.size
        if self.cvat_mask:
            mask_format = "cvat_mask"
        else:
            mask_format = "rle"
        slide_imgs, img_infos = one_task_slidewindow(self.s_width, self.s_height, self.overlap, image, mode="pil")
        for slide_name, slide_img in slide_imgs.items():
            bboxes, labels, scores = self.gen_box_prompts(Image.fromarray(slide_img))
            slide_results[slide_name] = self.predict_sam(slide_img, bboxes, scores, labels)
        if not self.sam_only:
            merge_anns = one_task_restore(img_infos, slide_results, self.s_width, self.s_height, self.confidence_score, (W, H), mask_format=mask_format)
        else:
            merge_anns = one_task_restore_semantic(img_infos, slide_results, self.s_width, self.s_height, self.confidence_score, (W, H), mask_format=mask_format)
        for merge_ann in merge_anns:
            total_masks.append(merge_ann["segmentation"])
            total_labels.append(merge_ann["category_id"])
            total_scores.append(merge_ann["score"])
            total_bboxes.append(merge_ann["bbox"])
        return {"bboxes": total_bboxes, "labels": total_labels, "scores":total_scores, "segmentations":total_masks}

def sam_inference(args, cvat_url: str, username: str, password: str, task_id: int, config_file: str, project_dir: str, confidence_score: float, slidewindow: bool, slidewindow_size: tuple, overlap: int, sam_only: bool):
    with make_client(host=cvat_url, credentials=(username, password)) as client:
        # client.api_client.set_default_header("Authorization", f"Token {token}")
        # client.config.status_check_period = 2
        task = client.tasks.retrieve(task_id)
        dataset = TaskDataset(client, task_id, load_annotations=False)
        SamInf = SamSegmentationFunction(args, config_file, project_dir, confidence_score, slidewindow, slidewindow_size, overlap, sam_only, cvat_mask=False)
        SamInf.inference_cvat(task, dataset, project_dir)
    print(f"Successfully do inference on CVAT Task {task_id}")
        
        