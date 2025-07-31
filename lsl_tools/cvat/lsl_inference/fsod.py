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
from typing import List

from PIL import Image
import torch
import numpy as np
from cvat_sdk import make_client
import cvat_sdk.models as models
import cvat_sdk.auto_annotation as cvataa

from lsl_tools.cvat.lsl_inference.utils import glip_cfg_setup, load_glip_model, gen_captions, unset_proxy
from lsl_tools.data.slide_windows import one_task_slidewindow
from lsl_tools.data.slide_windows_restore import one_task_restore
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.data.datasets.evaluation.box_aug import detect_bbox


np.float = np.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FsodDetectionFunction:
    def __init__(self, config_file: str, project_dir: str, confidence_score: float, slidewindow: bool, slidewindow_size: tuple, overlap: int) -> None:
        # Update the Cfg
        cfg = glip_cfg_setup(config_file, project_dir)
        # LOADING THE MODEL
        self.glip_model = load_glip_model(cfg, project_dir)
        # Captions and postive_map
        self.captions, self.positive_map_label_to_token, self.categories = gen_captions(cfg, project_dir)
        self.test_transform = build_transforms(cfg, is_train=False)
        # slidewindow
        self.slidewindow = slidewindow
        self.s_width, self.s_height = slidewindow_size
        self.overlap = overlap

        self.confidence_score = confidence_score
        unset_proxy()
    
    @property
    def spec(self) -> cvataa.DetectionFunctionSpec:
        # describe the annotations
        return cvataa.DetectionFunctionSpec(
            labels=[
                cvataa.label_spec(cat, i)
                for i, cat in self.categories.items()
            ]
        )
    
    def xyxy2xywh(self, bboxes):
        bboxes[..., 2] = bboxes[..., 2] - bboxes[..., 0]
        bboxes[..., 3] = bboxes[..., 3] - bboxes[..., 1]
        return bboxes
    
    def detect(self, context, image: Image.Image) -> List[models.LabeledShapeRequest]:
        # convert the input into a form the model can understand
        if not self.slidewindow:
            bboxes, labels, scores = self.predict(image)
        else:
            bboxes, labels, scores = self.predict_slidewindow(image)
        
        return [
            cvataa.rectangle(label, box)
            for box, label in zip(bboxes, labels)
        ]

    def predict(self, image: Image.Image):
        W, H = image.size

        with torch.no_grad():
            # run the ML model
            results = detect_bbox(
                        model=self.glip_model,
                        image=image,
                        transform=self.test_transform,
                        device=DEVICE,
                        captions=self.captions,
                        positive_map_label_to_token=self.positive_map_label_to_token
                    )
            results = results[0].resize((W, H))
            bboxes, labels, scores = self.filter_score(results)

        # convert the results into a form CVAT can understand
        return bboxes, labels, scores 
    
    def predict_slidewindow(self, image: Image.Image):
        slide_results = {}
        merge_bboxes = []
        merge_labels = []
        merge_scores = []
        W, H = image.size
        slide_imgs, img_infos = one_task_slidewindow(self.s_width, self.s_height, self.overlap, image, mode="pil")

        for slide_name, slide_img in slide_imgs.items():
            bboxes, labels, scores = self.predict(Image.fromarray(slide_img))
            if len(scores) == 0:
                continue
            slide_results[slide_name] = {"bboxes": self.xyxy2xywh(np.array(bboxes)), "labels": labels, "scores": scores, "segmentations": None}
        merge_anns = one_task_restore(img_infos, slide_results, self.s_width, self.s_height, self.confidence_score, (W, H))
        for merge_ann in merge_anns:
            bbox = merge_ann["bbox"]
            bbox[2] = bbox[2] + bbox[0]
            bbox[3] = bbox[3] + bbox[1]
            merge_bboxes.append(bbox)
            merge_labels.append(merge_ann["category_id"])
            merge_scores.append(merge_ann["score"])
        return merge_bboxes, merge_labels, merge_scores

    def filter_score(self, results):
        bboxes = results.bbox.cpu().detach().numpy()
        labels = results.get_field("labels").detach().cpu().numpy()
        scores = results.get_field("scores").detach().cpu().numpy()
        score_keep = (scores >= self.confidence_score)
        bboxes = bboxes[score_keep].tolist()
        labels = labels[score_keep].tolist()
        scores = scores[score_keep].tolist()
        return bboxes, labels, scores


def fsod_inference(cvat_url: str, username: str, password: str, task_id: int, config_file: str, project_dir: str, confidence_score: float, slidewindow: bool, slidewindow_size: tuple, overlap: int):
    with make_client(host=cvat_url, credentials=(username, password)) as client:
        # client.api_client.set_default_header("Authorization", f"Token {token}")
        # client.config.status_check_period = 2
        cvataa.annotate_task(client, task_id,
                             FsodDetectionFunction(config_file, project_dir, confidence_score, slidewindow, slidewindow_size, overlap),
                             allow_unmatched_labels=True)
        print(f"Successfully do inference on CVAT Task {task_id}")