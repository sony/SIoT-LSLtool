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
import argparse

import wget
from huggingface_hub import snapshot_download


import lsl_tools
lsl_path = os.path.abspath(os.path.dirname(lsl_tools.__file__))
model_root = f"{lsl_path}/pretrain_weights"


def get_parser():
    parser = argparse.ArgumentParser(description="Training Pipeline")
    parser.add_argument(
        "--task", default=None, choices=["fsod", "wsis", "fsis"], 
        help="lsl task mode")
    return parser.parse_args()

def download_model(model_path, url):
    if not os.path.exists(model_path):
        _ = wget.download(url, out=f"{model_path}")
        print(f"Download the pretrained model in {model_path}")
    else:
        print(f"The pretrained model has been downloaded before in {model_path}")

def download_fsod():
    if not os.path.exists(f"{model_root}/glip"):
        os.makedirs(f"{model_root}/glip")

    # download glip
    print("Download GLIP model")
    glip_path = f"{model_root}/glip/glip_tiny_model_o365_goldg_cc_sbu.pth"
    glip_url = "https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_tiny_model_o365_goldg_cc_sbu.pth"
    download_model(glip_path, glip_url)

    # download bert-base-uncased
    print("Download bert_base_uncased model")
    bert_base_uncased_dir = f"{model_root}/glip/bert-base-uncased"
    if not os.path.exists(bert_base_uncased_dir):
        snapshot_download(repo_id="google-bert/bert-base-uncased", local_dir=bert_base_uncased_dir)
    else:
        print(f"The pretrained model has been downloaded before in {bert_base_uncased_dir}")

def download_wsis():
    # fsod
    download_fsod()

    print("Download SAM model")
    if not os.path.exists(f"{model_root}/sam"):
        os.mkdir(f"{model_root}/sam")
    # download sam
    sam_path = f"{model_root}/sam/sam_vit_b_01ec64.pth"
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    download_model(sam_path, sam_url)

def download_fsis():
    # wsis
    download_wsis()

    if not os.path.exists(f"{model_root}/ImageNetPretrained"):
        os.mkdir(f"{model_root}/ImageNetPretrained")
    # download mb2
    print("Download MobileNetV2 model")
    mb2_path = f"{model_root}/ImageNetPretrained/mobilenet_v2-b0353104.pth"
    mb2_url = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
    download_model(mb2_path, mb2_url)

if __name__=="__main__":
    args = get_parser()
    if args.task == "fsod":
        download_fsod()
    elif args.task == "wsis":
        download_wsis()
    elif args.task == "fsis":
        download_fsis()