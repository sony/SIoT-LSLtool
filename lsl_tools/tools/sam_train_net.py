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
#

import os
import sys
dir_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, dir_path)
import os.path as opt
import json

from pycocotools.coco import COCO

from lsl_tools.sam.conf import get_parser
from lsl_tools.tools import lsl_args
from lsl_tools.sam.conf.defaults import get_default_cfg
from lsl_tools.sam.train_net import cfg2args
from lsl_tools.cli.utils import CfgDict
from lsl_tools.sam.engine.trainer import Training


def setup_sam_lsl(data_info, train_names, valid_name, test_name, task, iterations, conf_thresh, check=True):
    coco = COCO(data_info.labeled[train_names[0]].ann)
    cats = coco.loadCats(coco.getCatIds())
    cats_info = tuple([cat["name"] for cat in cats])
    num_train_images = len(coco.getImgIds())
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

    args = get_parser()
    cfg = get_default_cfg()
    if task == "fsis":
        config_file = opt.join(file_dir, "..", "config/sam/segm/sam.yaml")
    else:
        config_file = opt.join(file_dir, "..", "config/sam/segm/sam_only.yaml")
    cfg.merge_from_file(config_file)
    
    # mkdir output_dir
    output_dir = cfg.OUTPUT_DIR = opt.join(lsl_args.voc_tmp, train_names[0])
    # MODEL
    cfg.MODEL.NET = 'sam'  # choice in ["sam", "sam-l", "sam-adapter"]
    
    # DATASETS
    cfg.DATASETS.TRAIN = data_info["labeled"][train_names[0]].img
    cfg.DATASETS.VAL = data_info["labeled"][valid_name].img
    cfg.DATASETS.TRAIN_ANN = data_info["labeled"][train_names[0]].ann
    cfg.DATASETS.VAL_ANN = data_info["labeled"][valid_name].ann
    # cfg.DATASETS.PROMPT = val_prompts_path

    with open(cfg.DATASETS.TRAIN_ANN, "r") as fb:
        train_img_infos = json.load(fb)["images"]
    train_height = min([img_info["height"] for img_info in train_img_infos])
    train_width = min([img_info["width"] for img_info in train_img_infos]) 
    cfg.DATASETS.CROP_SIZE = (int(0.8 * train_width), int(0.8 * train_height))

    # SOLVER
    if cfg.MODEL.NET == 'sam':
        cfg.SOLVER.SAM_CKPT = opt.join(file_dir, "..", "pretrain_weights/sam/sam_vit_b_01ec64.pth")
    elif cfg.MODEL.NET == 'sam-l':
        cfg.SOLVER.SAM_CKPT = opt.join(file_dir, "..", "pretrain_weights/sam/sam_vit_l_0b3195.pth")
    if not os.path.exists(cfg.SOLVER.SAM_CKPT): raise Exception("No sam pretrained weight, please download it by download_pretrained_models.py")

    if lsl_args.mask_threshold != -1:
        cfg.SOLVER.THRESHOLD = lsl_args.mask_threshold

    if lsl_args.iter != -1:
        cfg.SOLVER.MAX_EPOCHS = int(lsl_args.iter // num_train_images)
        if lsl_args.iter % num_train_images > 0:
            cfg.SOLVER.MAX_EPOCHS+=1
        cfg.SOLVER.STEP = [int(0.6 * cfg.SOLVER.MAX_EPOCHS), int(0.8 * cfg.SOLVER.MAX_EPOCHS)]
    
    cfg.EXP_NAME = cfg.MODEL.NET
    best_checkpoint = f"{output_dir}/sam/Model/checkpoint_best.pth"
    if lsl_args.fs_base_model is not None and lsl_args.fs_resume:
        cfg.SOLVER.WEIGHT = lsl_args.fs_base_model
    elif os.path.exists(best_checkpoint):
        if check:
            while True:
                answer = input("The pre-trained SAM model already exists, whether to retrain it, enter Y or N: ")
                if answer.lower() == "n":
                    cfg.SOLVER.WEIGHT = best_checkpoint
                    if not lsl_args.fs_resume:
                        cfg.SOLVER.MAX_EPOCHS = 0
                    break
                elif answer.lower() == "y":
                    break
                else:
                    print("Input error, please enter Y or N")
        else:
            cfg.SOLVER.WEIGHT = best_checkpoint
            if not lsl_args.fs_resume:
                cfg.SOLVER.MAX_EPOCHS = 0
    if not cfg.MODEL.SAM_ONLY:
        val_prompts_path, _ = prepare_prompts(output_dir, data_info, train_names, valid_name, test_name, iterations, conf_thresh)
        cfg.DATASETS.PROMPT = val_prompts_path
    else:
        num_classes = len(cats_info)
        _ = prepare_prompts_sam_only(cfg, output_dir, data_info, test_name, num_classes)
    cfg.freeze()
    args = cfg2args(args, cfg)
    return args, cfg

def prepare_prompts(output_dir, data_info, train_names, valid_name, test_name, iterations, conf_thresh):
    skip_glip = True
    skip_train = False
    suffix_to_find = "model_best.pth"
    glip_weight_path =  opt.join(output_dir, suffix_to_find)
    val_prompts_path = f"{output_dir}/inference/val/bbox_background.json"
    if test_name:
        target_prompts_path = f"{output_dir}/inference/{test_name}/pseudo_label.json"
        if not os.path.exists(target_prompts_path):
            skip_glip = False
    else:
        target_prompts_path = None

    if not os.path.exists(glip_weight_path) or not os.path.exists(val_prompts_path):
        skip_glip = False
    else:
        while True:
            answer = input("The pre-trained GLIP model already exists, Whether to retrain it, enter Y or N: ")
            if answer.lower() == "y":
                skip_glip = False
                break
            elif answer.lower() == "n":
                skip_train = True
                break
            else:
                print("Input error, please enter Y or N")
    if not skip_glip:
        from lsl_tools.tools.glip_train_net import train_glip
        print("Training GLIP Model ...")
        train_glip(data_info, train_names, valid_name, test_name, iterations, conf_thresh, skip_train=skip_train)
        print("Training finished")
    return val_prompts_path, target_prompts_path

def train_sam(data_info, train_names, valid_name, test_name=None, task="fsis", iterations=100, conf_thresh=0.05):
    args, cfg = setup_sam_lsl(data_info, train_names, valid_name, test_name, task, iterations, conf_thresh)
    # print("\nTraining configure:")
    # print(cfg)
    Trainer = Training()
    Trainer.train(args)
    print("Training finished")   
    if test_name:
        # generate pseudo labels
        inference_pseudo_label(args, cfg, data_info, test_name, conf_thresh)

def sam_no_train(data_info, train_names, valid_name, test_name=None, iterations=100, conf_thresh=0.05, task="fsis", ):
    # print("\nTraining configure:")
    args, cfg = setup_sam_lsl(data_info, train_names, valid_name, test_name, task, iterations, conf_thresh, check=False)
    # print(cfg)
    Trainer = Training()
    Trainer.valid_base_sam(args)
      
    if test_name:
        # generate pseudo labels
        inference_pseudo_label(args, cfg, data_info, test_name, conf_thresh, no_train=True)

def inference_pseudo_label(args, cfg, data_info, test_name, conf_thresh, no_train=False):
    print("Generating pseudo labels ...")
    if not opt.exists(f"{cfg.OUTPUT_DIR}/inference/{test_name}"):
        os.mkdir(f"{cfg.OUTPUT_DIR}/inference/{test_name}")
    target_prompts_path = f"{cfg.OUTPUT_DIR}/inference/{test_name}/pseudo_label.json"
    target_img_dir = data_info["unlabeled"][test_name].img
    target_result_path = f"{cfg.OUTPUT_DIR}/inference/{test_name}/pseudo_mask_label.json"
    Trainer = Training()
    Trainer.inference_labels(args, target_img_dir, target_prompts_path, target_result_path, conf_thresh, no_train)
    print(f"Saving the Pseudo Mask Labels in {target_result_path}")

def prepare_prompts_sam_only(cfg, output_dir, data_info, test_name, num_classes):
    # val_prompts_path = f"{output_dir}/inference/val/bbox_background.json"
    from lsl_tools.sam.engine.train_mb2 import Mobilenetv2Training
    class_args = get_parser()
    cfg.freeze()
    class_args = cfg2args(class_args, cfg)
    class_args.base_lr = 1e-3
    class_args.ims_per_batch = 4
    class_args.max_epochs = 10
    class_args.step = [int(0.6 * class_args.max_epochs), int(0.8 * class_args.max_epochs)]
    class_args.num_classes = num_classes + 1
    ClassficationTrainer = Mobilenetv2Training()
    ClassficationTrainer.train(class_args, output_dir)
    if test_name:
        target_img_dir = data_info["unlabeled"][test_name].img
        target_prompts_root = f"{output_dir}/inference/{test_name}"
        target_prompts_path = f"{output_dir}/inference/{test_name}/pseudo_label.json"
        if not os.path.exists(target_prompts_root):
            os.makedirs(target_prompts_root)
        ClassficationTrainer.inference_label(class_args, target_img_dir, target_prompts_path)
        # gen_prompts_files(target_img_dir, target_prompts_path, val_ann_path, output_dir)
    else:
        target_prompts_path = None

    return target_prompts_path

if __name__ == "__main__":
    train_sam()
