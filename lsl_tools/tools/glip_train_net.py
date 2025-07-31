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

import json
import orjson
import os
import math
import sys
dir_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, dir_path)
from copy import deepcopy
import os.path as opt

from pycocotools.coco import COCO

from lsl_tools.tools import lsl_args
from lsl_tools.cli.utils import CfgDict
from lsl_tools.glip_lsl.finetune import train, test, tuning_highlevel_override
from lsl_tools.glip_lsl.glip_proposal import inference_labels
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.modeling.detector import build_detection_model


def del_background(ann_path):
    adjust = False  
    dataset = COCO(ann_path).dataset
    categories = dataset["categories"]

    annotations = dataset["annotations"]
    if categories[0]["name"] == "__background__":
        adjust = True
        categories = categories[1:]
    elif categories[0]["id"] == 0:
        adjust = True
        for annotation in annotations:
            annotation["category_id"]+=1
        for category in categories:
            category["id"]+=1

    dataset["categories"] = categories
    dataset["annotations"] = annotations
    return dataset, adjust

def get_nobackground_categories(out_dir):
    categories_dir = os.path.join(out_dir, "cate_info.json")
    with open(categories_dir, "r") as f:
        coco_categories = json.load(f)
    if coco_categories[0]["name"] == "__background__":
        coco_categories = coco_categories[1:]
    elif coco_categories[0]["id"] == 0:
        for category in coco_categories:
            category["id"]+=1
    return coco_categories

def revert_background(cfg, valid_json_path, pred_json_path, output_json_path):
    with open(valid_json_path, "r") as fp:
        source_data = json.load(fp)
        categories = source_data["categories"]

    no_bk_categories = get_nobackground_categories(cfg.OUTPUT_DIR)
    map_nobackground = {cat["name"]: cat["id"] for cat in no_bk_categories}
    bk_reversed_dict = {map_nobackground[category["name"]]: category["id"] for category in categories}
    with open(pred_json_path, "r") as fb:
        valid_preds = json.load(fb)
    for idx in range(len(valid_preds)):
        category_id = valid_preds[idx]["category_id"]
        valid_preds[idx]["category_id"] = bk_reversed_dict[category_id]
        valid_preds[idx]["id"] = idx
    
    # source_data["categories"] = coco_categories
    source_data["annotations"] = valid_preds
    with open(output_json_path, "w") as fc:
        json.dump(source_data, fc)

def setup_glip_lsl(data_info, train_names, valid_name, test_name, iterations):
    coco = COCO(data_info.labeled[train_names[0]].ann)
    cats = coco.loadCats(coco.getCatIds())
    cats_info = tuple([cat["name"] for cat in cats])
    num_train_images = len(coco.getImgIds())
    num_classes = len(cats_info)
    id2dir_map = None

    lsl_info = CfgDict({
        "data_info": data_info, "train_names": train_names,
        "valid_name": valid_name, "test_name": test_name,
        "cats_info": cats_info, "weak": False,
        "id2dir_map": id2dir_map,
    })

    file_dir = opt.dirname(__file__)
    dataset_root = opt.join(os.getcwd(), ".lsl")
    lsl_info["dataset_root"] = dataset_root
    os.environ["TORCH_EXTENSIONS_DIR"] = opt.join(file_dir, "..", "cache/torch_extensions")

    args = CfgDict()
    args.config_file = opt.join(file_dir, "..", "config/coco/coco_glip_finetune.yaml")
    args.skip_test = lsl_args.no_post_training_val
    args.local_rank = 0
    args.use_tensorboard = False

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.skip_optimizer_resume = not lsl_args.fs_resume
    args.evaluate_only_best_on_test = True
    args.push_both_val_and_test = True
    args.keep_testing = False
    args.skip_train = True
    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus
    cfg.merge_from_file(args.config_file)

    # mkdir output_dir
    output_dir = cfg.OUTPUT_DIR = opt.join(lsl_args.voc_tmp, train_names[0])
    if not opt.exists(output_dir):
        mkdir(output_dir)
    
    # generate the coco fmt json without background
    for data_name in [train_names[0], valid_name]:
        no_background_dataset, adjust = del_background(data_info["labeled"][data_name].ann)
        if adjust:
            ann_path = opt.join(output_dir, "_no_background.".join(os.path.basename(data_info["labeled"][data_name].ann).split(".")))
            with open(ann_path, "wb") as fo:
                fo.write(orjson.dumps(no_background_dataset))
            data_info["labeled"][data_name].ann = ann_path

    # generate the captions.txt
    with open(data_info["labeled"][train_names[0]].ann, "r") as fp:
        source_data = json.load(fp)
        categories = source_data["categories"]
        num_images = len(source_data["images"])
    num_classes = len(categories)
    captions = [i["name"] for i in categories]
    args.caption = ". ".join(captions)
    
    cfg.DATASETS.REGISTER.train.ann_file = data_info["labeled"][train_names[0]].ann
    cfg.DATASETS.REGISTER.train.img_dir = data_info["labeled"][train_names[0]].img
    cfg.DATASETS.REGISTER.val.ann_file = data_info["labeled"][valid_name].ann
    cfg.DATASETS.REGISTER.val.img_dir = data_info["labeled"][valid_name].img
    cfg.MODEL.ATSS.NUM_CLASSES = num_classes + 1
    cfg.MODEL.DYHEAD.NUM_CLASSES = num_classes + 1
    cfg.MODEL.FCOS.NUM_CLASSES = num_classes + 1
    cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = num_classes + 1
    if lsl_args.iter != -1:
        cfg.SOLVER.MAX_EPOCH = math.ceil(lsl_args.iter / num_images)
    cfg.DATASETS.FEW_SHOT = 0
    if lsl_args.fs_base_model is not None:
        cfg.MODEL.WEIGHT = lsl_args.fs_base_model
    else:
        cfg.MODEL.WEIGHT = opt.join(file_dir, "..", "pretrain_weights/glip/glip_tiny_model_o365_goldg_cc_sbu.pth")
        if not os.path.exists(cfg.MODEL.WEIGHT): raise Exception("No glip pretrained weight, please download it by download_pretrained_models.py")
    if num_train_images >= 10:
        cfg.DATASETS.GENERAL_COPY = 1
    else:
        cfg.DATASETS.GENERAL_COPY = int(10 // num_train_images)

    origin_cats = opt.join(output_dir, 'cate_info.json')
    with open(origin_cats, "wb") as fo:
        fo.write(orjson.dumps(cats))
    
    tuning_highlevel_override(cfg)
    return args, cfg, lsl_info


def train_glip(data_info, train_names, valid_name, test_name=None, iterations=100, conf_thresh=0.05, skip_train=True):
    glip_data_info = deepcopy(data_info)
    args, cfg, _ = setup_glip_lsl(glip_data_info, train_names, valid_name, test_name, iterations)
    # print("\nTraining configure:")
    # print(cfg)
    output_config_path = opt.join(cfg.OUTPUT_DIR, 'config.yml')
    print("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)

    suffix_to_find = "model_best.pth"
    weight_path =  opt.join(cfg.OUTPUT_DIR, suffix_to_find)
    args.skip_train = False
    if os.path.exists(weight_path) and not lsl_args.fs_resume:
        if skip_train: 
            while True:
                answer = input("The pre-trained GLIP model already exists, Whether to retrain it, enter Y or N: ")
                if answer.lower() == "y":
                    break
                elif answer.lower() == "n":
                    args.skip_train = True
                    break
                else:
                    print("Input error, please enter Y or N")

    train_cfg = cfg.clone()
    train_cfg.freeze()
    model = train(
        train_cfg,
        args, 
        args.local_rank, 
        args.distributed, 
        args.skip_train, 
        skip_optimizer_resume=args.skip_optimizer_resume,
        save_config_path=output_config_path)
    del model
    print("Training finished")   
    # remove previous visualization results
    # vis_dir = os.path.join(cfg['OUTPUT_DIR'], 'vis')
    # if opt.exists(vis_dir):
    #     shutil.rmtree(vis_dir)

    if not args.skip_test:
        # inference on the best model
        inference(args, cfg, weight_path)
        valid_json_path = data_info["labeled"][valid_name].ann
        pred_json_path = f"{cfg.OUTPUT_DIR}/inference/val/bbox.json"
        output_json_path = f"{cfg.OUTPUT_DIR}/inference/val/bbox_background.json"
        revert_background(cfg, valid_json_path, pred_json_path, output_json_path)

    if test_name:
        # generate pseudo labels
        inference_pseudo_label(args, cfg, data_info, test_name, conf_thresh)

def inference(args, cfg, weight_path):
    # load the best checkpointer
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    _ = checkpointer.load(weight_path, force=True)
    model.training = False
    model.eval()   
    # report valid map on training model
    print("Testing on validation dataset ...")
    test(cfg, model, args.distributed)

def inference_pseudo_label(args, cfg, data_info, test_name, conf_thresh):
    print("Generating pseudo labels ...")
    if not opt.exists(f"{cfg.OUTPUT_DIR}/inference/{test_name}"):
        os.mkdir(f"{cfg.OUTPUT_DIR}/inference/{test_name}")
    args.image_list_path = f"{cfg.OUTPUT_DIR}/{test_name}.txt"
    args.coco_fmt_output = f"{cfg.OUTPUT_DIR}/inference/{test_name}/pseudo_label.json"
    args.vis = None
    args.weak_cap = False
    test_images = os.listdir(data_info["unlabeled"][test_name].img)
    test_images = [opt.join(data_info["unlabeled"][test_name].img, img) for img in test_images]
    with open(args.image_list_path, "w") as f:
        f.write("\n".join(test_images))
        f.write("\n")
    test_cfg = cfg.clone()
    test_cfg.MODEL.ATSS.INFERENCE_TH = conf_thresh
    test_cfg.freeze()
    inference_labels(args, test_cfg)

if __name__ == "__main__":
    train_glip()
