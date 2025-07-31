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
import numpy as np
from label_studio_sdk import Client
from label_studio_ml.utils import get_image_local_path


class Import_LabelStudio(object):
    def __init__(self, LABEL_STUDIO_URL, API_KEY) -> None:
        self.LABEL_STUDIO_URL = LABEL_STUDIO_URL
        self.API_KEY = API_KEY
        
        #unset proxy
        try:
            del os.environ['http_proxy']
            del os.environ['https_proxy']
        except:
            print("No http proxy")
        self.ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
        self.ls.check_connection()

    def load_project(self, project_id):
        self.project = self.ls.get_project(project_id)

    def convert_coco_fmt(self, annotation, ann_type, img_id, width, height, category_ids_dict, use_polygons):
        if use_polygons:
            polygons = np.array(annotation["value"]["points"])
            polygons[..., 0] = polygons[..., 0] * width / 100
            polygons[..., 1] = polygons[..., 1] * height / 100
            min_x, max_x = min(polygons[..., 0]), max(polygons[..., 0])
            min_y, max_y = min(polygons[..., 1]), max(polygons[..., 1])
            bbox = [min_x, min_y, max_x-min_x, max_y-min_y]
            segmentation = polygons.reshape((1, -1)).tolist()
            label = annotation["value"]["polygonlabels"][0]
        else:
            min_x, min_y = annotation["value"]["x"], annotation["value"]["y"]
            b_width, b_height = annotation["value"]["width"], annotation["value"]["height"]
            bbox = [min_x*width/100, min_y*height/100, b_width*width/100, b_height*height/100]
            segmentation = []
            label = annotation["value"]["rectanglelabels"][0]
        category_id = category_ids_dict[label]
        return bbox, segmentation, category_id
    
    def get_img_path(self, raw_img_path):
        img_path = get_image_local_path(
                    raw_img_path,
                    label_studio_access_token=self.API_KEY,
                    label_studio_host=self.LABEL_STUDIO_URL
                )
        return img_path
    
    def get_categories(self):
        label_configs = self.project.parsed_label_config
        label_type = [value["type"] for key, value in label_configs.items()]
        if "PolygonLabels" in label_type:
            use_polygons = True
        else:
            use_polygons = False
        for key, config in label_configs.items():
            if use_polygons:
                if config["type"] == "PolygonLabels":
                    labels = config["labels"]
            else:
                if config["type"] == "RectangleLabels":
                    labels = config["labels"]
        categories = [{"id": idx+1, "name": label, "supercategory": label} for idx, label in enumerate(labels)]
        category_ids_dict =  {label: idx+1 for idx, label in enumerate(labels)}
        return categories, category_ids_dict, use_polygons
    
    def get_img_infos(self, task, annotations):
        sample_ann = annotations[0]
        width, height = sample_ann["original_width"], sample_ann["original_height"]
        img_path = self.get_img_path(task["data"]["image"])
        file_name = os.path.basename(img_path)
        return file_name, width, height

    def load_annotations(self):
        ann_id = 0
        image_infos = []
        annotations = []
        results = {}
        categories, category_ids_dict, use_polygons = self.get_categories()
        tasks = self.project.tasks
        sample_task = tasks[0]["data"]["image"]
        
        img_path = self.get_img_path(sample_task)
        img_root = os.path.dirname(img_path)
        
        for idx, task in enumerate(tasks):
            if len(task["annotations"]) == 0:
                continue
            anns = task["annotations"][-1]["result"]
            file_name, width, height = self.get_img_infos(task, anns)
            image_infos.append({
                "file_name": file_name,
                "height": height,
                "width": width,
                "id": idx
            })
            for ann in anns:
                ann_type = ann["type"]
                if use_polygons and ann_type == "rectanglelabels":
                    continue
                
                bbox, segmentation, category_id = self.convert_coco_fmt(ann, ann_type, idx, width, height, category_ids_dict, use_polygons)
                annotations.append({
                    "id": ann_id,
                    "image_id": idx,
                    "iscrowd": 0,
                    "bbox": bbox,
                    "segmentation": segmentation,
                    "area": bbox[2] * bbox[3],
                    "category_id": category_id
                })
                ann_id +=1
        results = {"images": image_infos, "annotations": annotations, "categories": categories}
        return results, img_root

    def export_project(self, labelstudio_id, output_root):
        self.load_project(labelstudio_id)
        results, img_root = self.load_annotations()
        output_path = f"{output_root}/{self.project.params['title']}.json"
        with open(output_path, "w") as f:
            json.dump(results, f)
        return img_root, len(results["images"]), output_path

if __name__=="__main__":
    labelstudio_id = 68
    LABEL_STUDIO_URL = 'http://localhost:8080'
    API_KEY = '88184212bba02066eb6cbf4a1baad388c9f22c33'
    labestudio_ci = Import_LabelStudio(LABEL_STUDIO_URL, API_KEY)
    labestudio_ci.export_project(labelstudio_id)





