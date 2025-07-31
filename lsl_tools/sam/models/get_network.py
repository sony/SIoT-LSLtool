# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
from functools import partial
from pathlib import Path
import urllib.request

import torch
import torch.nn as nn
from torchvision import models
from segment_anything.modeling import MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from ..models.image_encoder import ImageEncoderViT

def get_network(args, net, use_gpu=True, gpu_device = 0, distribution = True):
    """ return given network
    """
    device = torch.device('cuda', args.gpu_device)
    if net in ['sam', 'sam-adapter']:
        net = sam_model_registry['vit_b'](args,checkpoint=args.sam_ckpt).to(device)

    elif net == 'mobilenetv2':
        net = sam_model_registry['mobilenetv2'](args,checkpoint=None).to(device)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        #net = net.cuda(device = gpu_device)
        if distribution != 'none':
            net = torch.nn.DataParallel(net,device_ids=[int(id) for id in args.distributed.split(',')])
            net = net.to(device=gpu_device)
        else:
            net = net.to(device=gpu_device)

    return net

def build_sam_vit_b(args, checkpoint=None):
    return _build_sam(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def build_mb2(args, checkpoint=None):
    return _build_mb2(
        args,
        checkpoint=checkpoint,
    )

sam_model_registry = {
    "vit_b": build_sam_vit_b,
    'mobilenetv2': build_mb2,
}


def _build_sam(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = args.image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            args = args,
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    checkpoint = Path(checkpoint)
    if checkpoint.name == "sam_vit_b_01ec64.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_b_01ec64.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-B checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
        
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        if checkpoint.name == "checkpoint_best.pth":
            state_dict = state_dict["state_dict"]
        sam.load_state_dict(state_dict, strict = 212)
    return sam

def _build_mb2(args, checkpoint=None):
    num_classes = args.num_classes

    model = models.mobilenet_v2(pretrained=True, width_mult=1.0)
    last_channel = model.last_channel
    # replace mobilenet_v2  classifier layers
    classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(last_channel, num_classes),
    )
    model.classifier = classifier

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        # if checkpoint.name == "checkpoint_best.pth":
        #     state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict = True)
    return model