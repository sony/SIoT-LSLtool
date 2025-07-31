#!/usr/bin/env python3

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
import sys
import lsl_tools
# Set local_package_path to sys.path to affect lsl process package loading.
# Set to PYTHONPATH to affect package loading of child processes.
local_path = [os.path.abspath(os.path.dirname(lsl_tools.__file__)), os.path.abspath(os.path.dirname(lsl_tools.__file__) + '/packages')]
old_pypath = os.environ.get('PYTHONPATH', '')
path_sep = ';' if os.name == 'nt' else ':'
pretrain_weight_dir = os.path.join(local_path[0], "pretrain_weights")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTHONPATH'] = path_sep.join(local_path)
os.environ['PRETRAIN_WEIGHT_DIR'] = pretrain_weight_dir
for p in reversed(local_path):
    sys.path.insert(1, p)
if old_pypath:
    os.environ['PYTHONPATH'] += f'{path_sep}{old_pypath}'

import argparse
import yaml
import numpy as np

np.float = np.float64
np.int = np.int32

# clean logger 
import warnings
from transformers import logging

logging.set_verbosity_error()
logging.disable_default_handler()
warnings.filterwarnings("ignore")


def receive_command():
    parser = argparse.ArgumentParser("Labor saving tools")
    parser.add_argument(
        "command",
        type=str,
        choices=[
            "create", "import", "auto-label", 
            "export", "preview", "ls", 
            "rm", "rename", 
            "labelstudio-setup", "labelstudio-export", "labelstudio-ml-start",
            "cvat-setup", "cvat-export", "cvat-inference", "cvat-serverless",
            "slidewindows-setup", "split-data"
            ],
    )
    parser.add_argument(
        "-a",
        "--annotation",
        type=str,
        default=None,
        help="path to annotation file for labeled images",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default=None,
        help="path to labeled source images",
    )
    parser.add_argument(
        "-u",
        "--unlabeled-image",
        dest="unlabeled_image",
        type=str,
        default=None,
        help="target data to be labeled",
    )
    parser.add_argument(
        "-t",
        "--test",
        type=str,
        default=None,
        help="specify the target dataset to be labeled",
    )
    parser.add_argument(
        "-s",
        "--train",
        type=str,
        nargs='*',
        default=None,
        help="Train dataset used to label target images",
    )
    parser.add_argument(
        "-v",
        "--valid",
        dest="valid",
        type=str,
        default=None,
        help="valid dataset for evaluation",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="coco",
        help="data source labeled format",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        nargs='*',
        default=None,
        help="data source name",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=-1,                     # -1: adaptive training iteration
        help="training iterations",
    )
    parser.add_argument(
        "--valid-period",
        type=int,
        default=2000,
        help="valid period in training"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="confidence score of auto-labeling & preview",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=-1,
        help="threshold to filter the border of mask in fsis and fsis_segonly task",
    )
    parser.add_argument(
        "--task",
        default=None,
        type=str,
        choices=["fsod", "wsis", "fsis", "fsis_segonly"],
        help="lsl method to generate labels",
    )

    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="max number of images to preview",
    )
    parser.add_argument(
        "--reuse-prediction",          # load predictions.pth instead of do actual inference
        default=False,
        action="store_true",
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--reuse-prepared-dataset",    # skip dateset preparation, reuse existing dataset caches (speed up experiments)
        default=False,
        action="store_true",
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-post-training-val",      # skip post training validation
        default=False,
        action="store_true",
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--voc-tmp",
        default=os.path.join(os.getcwd(), ".lsl",),
        type=str,
        help=argparse.SUPPRESS)
    parser.add_argument(                # visualize detections
        "--show-prediction",
        default=False,
        action="store_true",
        help=argparse.SUPPRESS)
    parser.add_argument(                # suppress lsl configure by default
        "--show-config",
        default=False,
        action="store_true",
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--fs-base-model",              # fsod base model name
        default=None,
        type=str,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--fs-resume",                  # resume fsod training
        default=False,
        action="store_true",
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--url",                  # Label Studio URL
        default=None,
        type=str,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--token",                  # Label Studio Token
        default=None,
        type=str,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--cvat-username",                  # CVAT user name
        default=None,
        type=str,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--cvat-password",                  # CVAT password
        default=None,
        type=str,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--labelstudio-name",                  # LabelStudio Task name
        default=None,
        type=str,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--cvat-name",                  # CVAT Task name
        default=None,
        type=str,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--labelstudio-project-id",                  # Label Studio Project ID
        default=None,
        type=str,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--cvat-task-id",                  # CVAT Task ID
        default=None,
        type=int,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--cvat-serverless-image",                  # Base docker image name of lsl cvat serverless function
        default="zerorpc_demo:v1",
        type=str,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--cvat-orgname",           # CVAT Orgnization short name
        default=None,
        type=str,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--labelstudio-ml-port",                  # Label Studio ML Port
        default=12290,
        type=int,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--slidewindow",                  # Slide windows control
        default=False,
        action="store_true",
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--slidewindow-size",                  # (width, height)the cropped image size after slide windows
        default=(1024, 1024),
        type=int,
        nargs="+",
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--overlap",                  # the overlap of the sliding windows and should be bigger than the height and width of instances
        default=150,
        type=int,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--object-threshold",           # if IoU between cropped object part and original object lower than the value, it will be discarded
        default=0.2,
        type=float,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--split-ratio",           # split ratio for training set, e.g. 0.9
        default=0.9,
        type=float,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-path",           # output json path
        default=None,
        type=str,
        help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    # for user case, muting package warning info
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from lsl_tools.cli.data_source import DataManager
    from lsl_tools.cli.data_worker import DataWorker
    from lsl_tools.cli.factory import LSLProcess

    proj = LSLProcess(DataManager, DataWorker)
    args = receive_command()
    if args.command == 'auto-label' and args.show_config:
        print(f'\nLSL configure:\n{yaml.dump(vars(args))}\n')
    proj(args)

if __name__ == "__main__":
    main()
