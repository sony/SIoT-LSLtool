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
import glob
import os
import sys
import json
import time
import math
from random import shuffle
import shutil

from lsl_tools.tools import lsl_args
from lsl_tools.cli.utils.config_dict import CfgDict
from collections import defaultdict
from lsl_tools.tools.slidewindow import get_slidewindow
from lsl_tools.labelstudio.export_json import LSL_LabelStudio
from lsl_tools.labelstudio.labelstudio_converter import convert2labelstudio
from cvat.export_lsl import LSL_CVAT


class DataWorker:
    """ Interface of models and the lsl tool

    model should support coco format data training and train procedure begins with
    annotation file and image directories, which is the interface between DataWorker
    and models should be (ann_file, img_dir)
    """

    def __init__(self, method="transfer"):
        self.type = method
        self.work_dir = os.getcwd()
        self.ann_file = {}
        self.img_dir = {}
        self.unlabeled_data = None
        self.test_name = None
        self.training_info = None
    
    def work_wsis(self, data_info, test_name, train_names, valid_name, iterations, conf_thresh):
        """ the boxinstseg process of pseudo data generation

        Separate to two steps:
        1. weakly supervise training and evaluation

        args:
            data_info (CfgDict):   labeled and unlabeled data info in lsl .config file
            test_name (str):     dataset to be labeled
            train_names (list):   training dataset to train a model
            valid_name (float):    validation data set for train evaluation
            iterations (int):      train iteration of wetectron model
            conf_thresh (float):   not used in weakly learning setting
        """
        from lsl_tools.tools.sam_train_net import sam_no_train
        print("Training model ...")
        # Add GLIP + SAM
        sam_no_train(data_info, train_names, valid_name, test_name, iterations, conf_thresh)
        print("Training finished")

    def work_glip(self, data_info, test_name, train_names, valid_name, iterations, conf_thresh):
        """ the boxinstseg process of pseudo data generation

        Separate to two steps:
        1. weakly supervise training and evaluation

        args:
            data_info (CfgDict):   labeled and unlabeled data info in lsl .config file
            test_name (str):     dataset to be labeled
            train_names (list):   training dataset to train a model
            valid_name (float):    validation data set for train evaluation
            iterations (int):      train iteration of wetectron model
            conf_thresh (float):   not used in weakly learning setting
        """
        from lsl_tools.tools.glip_train_net import train_glip
        print("Training model ...")
        train_glip(data_info, train_names, valid_name, test_name, iterations, conf_thresh)
    
    def work_sam(self, data_info, test_name, train_names, valid_name, task, iterations, conf_thresh):
        """ the sam process of pseudo data generation

        args:
            data_info (CfgDict):   labeled and unlabeled data info in lsl .config file
            test_name (str):     dataset to be labeled
            train_names (list):   training dataset to train a model
            valid_name (float):    validation data set for train evaluation
            iterations (int):      train iteration of wetectron model
            conf_thresh (float):   not used in weakly learning setting
        """
        from lsl_tools.tools.sam_train_net import train_sam
        print("Training model ...")
        train_sam(data_info, train_names, valid_name, test_name, task, iterations, conf_thresh)
    
    def preview_coco_label(self, data_info, train_names, test_name, task, conf_thresh):
        from lsl_tools.tools.preview_coco import visualization_coco
        unlabeled_data_info = data_info.unlabeled[test_name[0]]
        unlabel_img_dir = unlabeled_data_info.img
        train_set_name = train_names[0]
        results_coco_path = f"{lsl_args.voc_tmp}/{train_set_name}/inference/{test_name[0]}"
        output_dir = f"{results_coco_path}/vis"
        if task in ["fsis", "fsis_segonly", "wsis"]:
            results_coco_path = f"{results_coco_path}/pseudo_mask_label.json"
        else:
            results_coco_path = f"{results_coco_path}/pseudo_label.json"
        print("start preview ... ")
        visualization_coco(unlabel_img_dir, results_coco_path, output_dir, conf_thresh)
        print(f"preview finished, the output images has been saved in {output_dir}")

    def parse_train_info(self, data_info, test_name, train_names, valid_ratio=0.2):
        for src in train_names:
            try:
                self.ann_file[src] = data_info.labeled[src].ann
                self.img_dir[src] = data_info.labeled[src].img
            except KeyError:
                print(f"Data source {src} doesn't exist, please import first or rename it")
                exit(0)
        self.unlabeled_data = data_info.unlabeled[test_name].img

        self.test_name = test_name
        self.training_info = CfgDict({
            "from_source": train_names,
            "valid_ap": -1,
            "validation_ratio": valid_ratio,
            "date": -1,
        })

    def write_annotation_file(self, output_info, cate_info):

        data_dict = {"images": [], "annotations": [], "categories": cate_info}
        for item in output_info:
            data_dict["images"].append(item["img_info"])
            data_dict["annotations"].extend(item["bboxes"])
        save_dir = os.path.join(self.work_dir, self.test_name)
        save_file = os.path.join(save_dir, "pseudo_label.json")
        with open(save_file, "w") as fo:
            json.dump(data_dict, fo)
        print(f"Generated pseudo label has been saved in \"{save_file}\"")

    def get_data_categories_info(self, ann_file):
        cate = self.read_json_file(ann_file)["categories"]
        num_class = len(cate)
        class_name = [item["name"] for item in cate]
        return num_class, class_name, cate
    
    def slidewindow_data(self, label_cmds, data_info):
        test_name = label_cmds.test
        train_names = label_cmds.train
        valid_name = label_cmds.valid
        SlideWindow = get_slidewindow(label_cmds)
        data_info, cropped_test_name = SlideWindow.prepare_dataset(data_info, train_names, valid_name, test_name)
        label_cmds.test = cropped_test_name
        return data_info, label_cmds
    
    def slidewindow_restore(self, label_cmds, cropped_label_cmds, data_info, task_name):
        test_name = label_cmds.test
        train_names = label_cmds.train
        cropped_test_name = cropped_label_cmds.test
        pred_dir = f"{lsl_args.voc_tmp}/{train_names[0]}/inference/{cropped_test_name}"
        output_dir = f"{lsl_args.voc_tmp}/{train_names[0]}/inference/{test_name}"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if task_name in ["fsis", "fsis_segonly", "wsis"]:
            ann_file = "pseudo_mask_label.json"
        else:
            ann_file = "pseudo_label.json"
        if task_name == "fsis_segonly":
            semantic = True
        else:
            semantic = False
        SlideWindow = get_slidewindow(label_cmds)
        SlideWindow.restore(
            img_dir=data_info["unlabeled"][test_name].img,
            pred_ann_path=f"{pred_dir}/{ann_file}",
            output_dir=f"{output_dir}/{ann_file}",
            score_threshold=label_cmds.conf,
            semantic=semantic)
        shutil.rmtree(pred_dir)
        return data_info, label_cmds

    @staticmethod
    def read_json_file(file_path):
        with open(file_path, "r") as fo:
            ann = json.load(fo)
        return ann

    @staticmethod
    def write_json_file(data, file_path):
        with open(file_path, "w") as fo:
            json.dump(data, fo)

    @staticmethod
    def single_ann_split(anno_file, split):
        ann_dir = os.path.dirname(anno_file)
        ann_data = DataWorker.read_json_file(anno_file)
        shuffle(ann_data["images"])
        split_point = int((1 - split) * len(ann_data["images"]))

        train_data = {
            "images": ann_data["images"][:split_point],
            "categories": ann_data["categories"],
        }
        train_img_id = [item["id"] for item in train_data["images"]]
        train_ann = [item for item in ann_data["annotations"] if item["image_id"] in train_img_id]
        train_data["annotations"] = train_ann
        train_ann_file = os.path.join(ann_dir, "lsl_train.json")
        DataWorker.write_json_file(train_data, train_ann_file)

        valid_data = {
            "images": ann_data["images"][split_point:],
            "categories": ann_data["categories"],
        }
        valid_img_id = [item["id"] for item in valid_data["images"]]
        valid_ann = [item for item in ann_data["annotations"] if item["image_id"] in valid_img_id]
        valid_data["annotations"] = valid_ann
        valid_ann_file = os.path.join(ann_dir, "lsl_valid.json")
        DataWorker.write_json_file(valid_data, valid_ann_file)
        return train_ann_file, valid_ann_file

    @staticmethod
    def export_ann_data(test_names, output_dir, image_dirs=None):
        temp = {
            "images": [],
            "categories": [],
            "annotations": [],
        }
        for target in test_names:
            try:
                data = DataWorker.read_json_file(f"{lsl_args.voc_tmp}/{target}/pseudo_label.json")
            except FileNotFoundError:
                print("No such auto-labeled annotation files")
                exit(0)

            cat_id2name = {}
            for cat in data["categories"]:
                cat_id2name[cat["id"]] = cat["name"]

            for img in data["images"]:
                img["gid"] = "{}_{}".format(target, img["id"])
                img["file_name"] = img["file_name"]

            for ann in data["annotations"]:
                ann["img_gid"] = "{}_{}".format(target, ann["image_id"])
                ann["cat_gid"] = "{}_{}".format(target, ann["category_id"])
                ann["cat_name"] = cat_id2name[ann["category_id"]]
            for cat in data["categories"]:
                cat["gid"] = "{}_{}".format(target, cat["id"])

            temp["images"].extend(data["images"])
            temp["annotations"].extend(data["annotations"])
            temp["categories"].extend(data["categories"])

        img_to_ann = defaultdict(list)
        for ann in temp["annotations"]:
            img_to_ann[ann["img_gid"]].append(ann)

        # merge the image_id if anns
        ann_list = []
        for i, img in enumerate(temp["images"]):
            img["id"] = i
            anns = img_to_ann[img["gid"]]
            for ann in anns:
                ann["image_id"] = i
            ann_list.extend(anns)
        temp["annotations"] = ann_list

        # merge the categories field
        name_set = set()
        for cat in temp["categories"]:
            name_set.add(cat["name"])

        cats = list(name_set)
        cat_dic_list = []
        for i, cat in enumerate(cats):
            cat_dict = {
                "supercategory": "",
                "id": i + 1,
                "name": cat,
            }
            cat_dic_list.append(cat_dict)
        temp["categories"] = cat_dic_list

        cat_name2id = {}
        for cat in temp["categories"]:
            cat_name2id[cat["name"]] = cat["id"]

        for ann in temp["annotations"]:
            ann["category_id"] = cat_name2id[ann["cat_name"]]

        pre_fix = ""
        for name in test_names:
            pre_fix += f"{name}_"
        fname = f"{output_dir}/{pre_fix}pseudo_label.json"
        DataWorker.write_json_file(temp, fname)
        print(
            "The labeled json file has been exported to\n",
            f"  {fname}"
        )
    
    @staticmethod
    def split_coco_data(input_path, output_path, split_ratio):
        split_coco_args = f' --input {input_path}'
        split_coco_args += f' --output {output_path}'
        split_coco_args += f' --split {split_ratio}'
        split_coco_script = os.path.normpath(f'{lsl_args.package_top}/tools/split_coco.py')
        split_coco_cmd = f'{sys.executable} {split_coco_script} {split_coco_args}'
        assert os.system(split_coco_cmd) == 0, f'Fail to split annotation file in {input_path}'

    @staticmethod
    def adjust_anns_for_labelstudio(coco_json_path):
        with open(coco_json_path, "r") as fa:
            coco_results = json.load(fa)
        annotations = coco_results["annotations"]
        adjust_anns = []
        for annotation in annotations:
            if annotation["segmentation"] == []:
                del annotation["segmentation"]
            adjust_anns.append(annotation)
        coco_results["annotations"] = adjust_anns
        adjust_coco_json_path = coco_json_path.replace(".json", "_adjust.json")
        with open(adjust_coco_json_path, "w") as fb:
            json.dump(coco_results, fb)
        return adjust_coco_json_path

    @staticmethod
    def export_to_labelstudio(config_state, source_name, test_names, project_name, output_dir, task_name, image_dirs=None):
        labelstudio_url = config_state.labelstudio.url
        api_key = config_state.labelstudio.token

        for test_name in test_names:
            img_dir = config_state.data_source.unlabeled[test_name].img
            if task_name in ["fsis", "fsis_segonly"]:
                coco_json_path = f"{lsl_args.voc_tmp}/{source_name}/inference/{test_name}/pseudo_mask_label.json"
            else:
                coco_json_path = f"{lsl_args.voc_tmp}/{source_name}/inference/{test_name}/pseudo_label.json"
                coco_json_path = DataWorker.adjust_anns_for_labelstudio(coco_json_path)
            labelstudio_output = f"{output_dir}/.lsl/{source_name}/inference/{test_name}"
            convert2labelstudio(coco_json_path, labelstudio_output) # convert coco json to labelstudio json

            labelstudio_json = f"{labelstudio_output}/preudo_label_labelstudio.json"
            labelstudio_xml = f"{labelstudio_output}/preudo_label_labelstudio.label_config.xml"
            print(f"Label Studio json saved in {labelstudio_output}/preudo_label_labelstudio.json")

            #unset proxy
            try:
                del os.environ['http_proxy']
                del os.environ['https_proxy']
            except:
                print("No http proxy")
            LSL_LabelStudio(labelstudio_url, api_key).export_labelstudio(labelstudio_json, labelstudio_xml, project_name, img_dir)
            print("")
            print("*" * 50)
            print(f"Successfully export LSL project: {test_name} to Label Studio project: {project_name} ")
            print("*" * 50)

    @staticmethod
    def labelstudio_ml_run(args, config_state, source_name, labelstudio_ml_port, confidence_score):
        labelstudio_url = config_state.labelstudio.url
        labelstudio_ml_port = args.labelstudio_ml_start.labelstudio_ml_port
        api_key = config_state.labelstudio.token
        task_name = args.task
        slidewindow = args.labelstudio_ml_start.slidewindow
        width, height=args.labelstudio_ml_start.slidewindow_size
        overlap=args.labelstudio_ml_start.overlap
        
        labelstudio_ml_args = f' --port {labelstudio_ml_port}'
        labelstudio_ml_args += f' --labelstudio-url {labelstudio_url}'
        labelstudio_ml_args += f' --token {api_key}'
        
        if task_name == "fsod": # GLIP
            labelstudio_ml_args += f' --project-dir {lsl_args.voc_tmp}/{source_name}'
            labelstudio_ml_args += f' --config-file {lsl_args.package_top}/config/coco/coco_glip_finetune.yaml'
            labelstudio_ml_args += f' --conf {confidence_score}'      
            lsl_project_name = "lsl_glip" 
        elif task_name in ["fsis", "fsis_segonly"]: # SAM
            labelstudio_ml_args += f' --project-dir {lsl_args.voc_tmp}/{source_name}'
            if task_name == "fsis":
                labelstudio_ml_args += f' --config-file {lsl_args.package_top}/config/sam/segm/sam.yaml'
            else:
                labelstudio_ml_args += f' --config-file {lsl_args.package_top}/config/sam/segm/sam_only.yaml'
                labelstudio_ml_args += f' --sam-only True' 
            lsl_project_name = "lsl_sam" 
        
        if slidewindow:
            labelstudio_ml_args += f' --slidewindow'
            labelstudio_ml_args += f' --slidewindow-size {width} {height}'
            labelstudio_ml_args += f' --overlap {overlap}'

        labelstudio_ml_script = os.path.normpath(f'{lsl_args.package_top}/labelstudio/label-studio-ml-backend/label_studio_ml/lsl_backends/{lsl_project_name}/_wsgi.py')
        labelstudio_ml_cmd = f'{sys.executable} {labelstudio_ml_script} {labelstudio_ml_args}'
        assert os.system(labelstudio_ml_cmd) == 0, f'Fail to start Label Studio ML for {task_name} task'

    @staticmethod
    def export_to_cvat(config_state, source_name, test_names, cvat_task_name, output_dir, task_name):
        cvat_url = config_state.cvat.url
        # api_key = config_state.cvat.token
        username = config_state.cvat.username
        password = config_state.cvat.password
        org_name = config_state.cvat.org_name

        for test_name in test_names:
            img_dir = config_state.data_source.unlabeled[test_name].img
            if task_name in ["fsis", "fsis_segonly"]:
                coco_json_path = f"{lsl_args.voc_tmp}/{source_name}/inference/{test_name}/pseudo_mask_label.json"
            else:
                coco_json_path = f"{lsl_args.voc_tmp}/{source_name}/inference/{test_name}/pseudo_label.json"

            #unset proxy
            try:
                del os.environ['http_proxy']
                del os.environ['https_proxy']
            except:
                print("No http proxy")
            LSL_CVAT(cvat_url, username, password, org_name).export_to_cvat(img_dir, coco_json_path, cvat_task_name)
            print("")
            print("*" * 50)
            print(f"Successfully export LSL project: {test_name} to Label Studio project: {cvat_task_name} ")
            print("*" * 50)
    
    @staticmethod
    def infer_cvat(args, config_state, source_name, task_id, confidence_score):
        cvat_url = config_state.cvat.url
        username = config_state.cvat.username
        password = config_state.cvat.password
        project_dir =  f"{lsl_args.voc_tmp}/{source_name}"
        
        task_name = args.task
        slidewindow = args.cvat_inference.slidewindow
        width, height = args.cvat_inference.slidewindow_size
        overlap = args.cvat_inference.overlap
        
        if task_name == "fsod": # GLIP
            from lsl_tools.cvat.lsl_inference.fsod import fsod_inference
            config_file = f"{lsl_args.package_top}/config/coco/coco_glip_finetune.yaml"
            fsod_inference(cvat_url, username, password, task_id, config_file, project_dir, confidence_score, slidewindow, (width, height), overlap)
        elif task_name in ["fsis", "fsis_segonly"]: # SAM
            from lsl_tools.cvat.lsl_inference.sam import sam_inference
            if task_name == "fsis":
                sam_only = False
                config_file = f"{lsl_args.package_top}/config/sam/segm/sam.yaml"
            else:
                sam_only = True
                config_file = f"{lsl_args.package_top}/config/sam/segm/sam_only.yaml"
            args.mask_threshold = args.cvat_inference.mask_threshold
            sam_inference(args, cvat_url, username, password, task_id, config_file, project_dir, confidence_score, slidewindow, (width, height), overlap, sam_only)
        else:
            print("CVAT inference function is available for sam, sam_only and fsod task")
    
    @staticmethod
    def build_cvat_serverless(args, source_name, cvat_serverless_image, confidence_score):
        project_dir =  f"{lsl_args.voc_tmp}/{source_name}"
        
        task_name = args.task
        slidewindow = args.cvat_serverless.slidewindow
        width, height = args.cvat_serverless.slidewindow_size
        overlap = args.cvat_serverless.overlap
        
        # Add docker catch shell path
        os.environ["PATH"] =f"{lsl_args.package_top}/cvat/lsl_ml_severless" + ":" + os.environ.get('PATH', '')

        assert task_name in ["fsod", "fsis", "fsis_segonly"], f"LSL CVAT serverless function has not supported mode:{task_name}"
        # sam no glip
        if task_name == "fsis_segonly":
            task_dir = "fsis"
        else:
            task_dir = task_name
        # build function yaml
        from lsl_tools.cvat.lsl_ml_severless.build_function import build_yaml
        build_yaml(project_dir, task_dir, cvat_serverless_image)

        # get zerorpc port
        from lsl_tools.cvat.lsl_ml_severless.get_zerorpc_port import getPort
        zerorpc_port = getPort(f'{lsl_args.package_top}/cvat/lsl_ml_severless/{task_dir}/zerorpc_port.txt')

        # run zerorpc service
        import socket
        import subprocess
        from pathlib import Path

        severless_script = os.path.normpath(f'{lsl_args.package_top}/cvat/lsl_ml_severless/{task_dir}/zerorpc_service.py')
        severless_args = f' --project-dir {project_dir}'
        severless_args += f' --port {zerorpc_port}'
        severless_args += f' --conf {confidence_score}'
        if slidewindow:
            severless_args += f' --slidewindow'
            severless_args += f' --slidewindow-size {width} {height}'
            severless_args += f' --overlap {overlap}'
        
        dataset_name = Path(project_dir).parent.parent.stem
        deploy_script = f"nuctl deploy lsl_{task_name}_{dataset_name}" 
        deploy_args = " --project-name cvat --runtime python --handler main:handler --platform local" 
        deploy_args += f" --path {lsl_args.package_top}/cvat/lsl_ml_severless/{task_dir}"
        deploy_args += f" --file {lsl_args.package_top}/cvat/lsl_ml_severless/{task_dir}/function.yaml"

        
        if task_name in ["fsis", "fsis_segonly"]:
            if task_name == "fsis": # SAM
                config_file = f"{lsl_args.package_top}/config/sam/segm/sam.yaml"
            else: # SAM Only
                config_file = f"{lsl_args.package_top}/config/sam/segm/sam_only.yaml"
                severless_args += f' --sam-only True'
            severless_args += f' --config-file {config_file}'
            severless_args += f' --mask-threshold {args.cvat_serverless.mask_threshold}'
            
        elif task_name == "fsod": # FSOD
            config_file = f"{lsl_args.package_top}/config/coco/coco_glip_finetune.yaml"

        severless_args += f' --config-file {config_file}'
        severless_cmd = f'{sys.executable} {severless_script} {severless_args}'

        deploy_cmd = f'{deploy_script} {deploy_args}'

        subprocess.Popen(severless_cmd, shell=True)

        total_times = 0
        while True:
            try:
                # listen the zerorpc port to confirm zerorpc service has been started
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                s.connect(("127.0.0.1", int(zerorpc_port)))
                break
            except:
                time.sleep(5)
                total_times+=1
                assert total_times < 50, f"Fail to start zerorpc service for {task_name}"

        assert os.system(deploy_cmd) == 0, f"Fail to deploy serverless docker image for {task_name}"

        while True:
            1

        

 