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
import json
import random
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser("Split COCO format dataset by a splitting ratio")
    parser.add_argument('--input', required=True, help='input json name')
    parser.add_argument('--output', required=True, help='output json name')
    parser.add_argument('--split', required=True, type=float, help='split ratio, e.g. 0.9')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    with open(args.input) as f:
        src = json.load(f)

    images = src['images']
    annotations = src['annotations']
    categories = src['categories']
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    split = float(args.split)
    # random.seed(0)
    random.shuffle(images)
    n_imgs = len(images)
    output_stem = Path(args.input).stem
    for i, range in enumerate([[0, split], [split, 1.0]]):
        start, stop = [int(n_imgs * x) for x in range]
        images_split = images[start:stop]
        image_ids = set()
        [image_ids.add(img['id']) for img in images_split]
        annotations_split = [anno for anno in annotations if anno['image_id'] in image_ids]
        output = f'{args.output}/{output_stem}_{i+1}_{len(images_split)}.json'
        print(f'Split {i+1}: {len(images_split)} images, {len(images_split)} annotations -> {str(output)}')
        with open(output, 'w') as f:
            json.dump({ 'categories':categories, 'images': images_split, 'annotations': annotations_split }, f)



