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
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import os.path as opt
import glob
import numpy as np
from lsl_tools.data.visualize import COLORS
from lsl_tools.tools import lsl_args
from PIL import Image


def preprocess_annotation(ann_file):
    target = ET.parse(ann_file).getroot()

    boxes_ls = []
    names_ls = []
    TO_REMOVE = 1

    for obj in target.iter("object"):
        name = obj.find("name").text.lower().strip()
        names_ls.append(name)
        bb = obj.find("bndbox")
        box = [
            bb.find("xmin").text,
            bb.find("ymin").text,
            bb.find("xmax").text,
            bb.find("ymax").text,
        ]
        bndbox = tuple(
            map(lambda x: x - TO_REMOVE, list(map(int, box)))
        )
        boxes_ls.append(bndbox)
    res = {
        "boxes": boxes_ls,
        "names": names_ls,
    }
    return res


def draw(img, anns):
    boxes, class_names = anns["boxes"], anns["names"]
    for i in range(len(boxes)):
        box = boxes[i]
        name = class_names[i]

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (COLORS[i] * 255).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

        text = name
        txt_color = (0, 0, 0) if np.mean(COLORS[i]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]

        txt_bk_color = (COLORS[i] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def draw_ann(img_file, ann_file, save_to):
    img = cv2.imread(img_file)
    ann = preprocess_annotation(ann_file)
    img = draw(img, ann)
    # img = img[..., ::-1]
    # Image.fromarray(img).show()
    file_name = opt.join(save_to, opt.basename(img_file))
    cv2.imwrite(file_name, img)