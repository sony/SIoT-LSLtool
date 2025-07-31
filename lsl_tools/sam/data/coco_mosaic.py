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
from typing import Tuple
from PIL import Image, ImageDraw
from copy import deepcopy
import random

import cv2
import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import trange
from pycocotools.coco import COCO
import pycocotools.mask as pymask
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

from ..utils.utils import random_click


class COCODataset(Dataset):
    def __init__(
        self, args, ann_file, img_dir, mode = 'Training', prompt = 'click'
    ):
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.padding = args.padding
        self.resize_size = args.resize_size
        self.padding_border = (512, 512)
        self.using_mosaic = args.mosaic
        self.mosaic_border = (-512, -512)
        self.img_dir = img_dir
        self.noised_bbox = args.noised_bbox
        self.noised_ratio = [i**0.5 / 2 for i in args.noised_ratio]
        self.sam_only = args.sam_only

        self.points = None 
        self.coco = COCO(ann_file)
        ids = list(self.coco.imgs.keys())
        self.ids = self._preprocess(ids)
        self.dataset_sample_index = list(range(len(self.ids)))
        self.aug_transforms = A.Compose([
            A.RandomCrop(width=args.crop_size[0], height=args.crop_size[1], p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # A.RandomBrightnessContrast(p=0.2)
            # A.OneOf([
            #     A.MotionBlur(p=0.1),
            #     A.MedianBlur(blur_limit=3, p=0.1),
            #     A.Blur(blur_limit=3, p=0.1),
            #     ], p=0.3),
            # A.OneOf([
            #     A.Sharpen(p=0.2),
            #     A.Emboss(p=0.1),
            #     A.RandomBrightnessContrast(p=0.2),
            #     ], p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.7)
        ], bbox_params=A.BboxParams(format='coco', min_area=10, min_visibility=0.25, label_fields=['category_id']))
        # self.aug_transforms.is_check_args = False

    def __getitem__(self, index):
        point_label = 1
        inout = 1
        if self.using_mosaic and self.mode == "Training" and random.random() < 0.5:
            img, image_id, name, bboxes, masks, categories = self.get_mosaic_sample(index)
        else:
            img, image_id, name, bboxes, masks, categories = self.get_sample(index)

        oldh, oldw  = img.shape[:-1]
        if self.resize_size is not None and not self.using_mosaic:
            newh, neww = self.get_preprocess_shape(oldh, oldw, self.resize_size)
        elif self.padding and max(oldh, oldw) < self.img_size:
            newh, neww = oldh, oldw
        else:
            newh, neww = self.get_preprocess_shape(oldh, oldw, self.img_size)

        padt = int((self.img_size - newh)/2)
        padl = int((self.img_size - neww)/2)
        padb = self.img_size - newh - padt
        padr = self.img_size - neww - padl

        img = self.transform_img(img, newh, neww, padt, padl, padb, padr)
        image_meta_dict = {'filename_or_obj':name}
        if self.mode == 'Test':
            return {
            'image':img,
            'label': [0, 0, 0, 0],
            'image_meta_dict':image_meta_dict,}
        else:
            transform_masks = [self.transform_mask(mask*255, newh, neww, padt, padl, padb, padr) for mask in masks]

        if len(masks) != 0:
            bboxes[..., 0] = bboxes[..., 0] + padl
            bboxes[..., 1] = bboxes[..., 1] + padt
            bboxes[..., 2] = bboxes[..., 2] + padl
            bboxes[..., 3] = bboxes[..., 3] + padt

        if self.prompt == 'click':
            pt = []
            no_gt = False
            masks = sum(transform_masks)

            # gen postive points
            if not isinstance(masks, int):
                # masks = torch.zeros((1, self.img_size, self.img_size))
                masks = torch.zeros((1, self.img_size, self.img_size))
                clear_transform_masks = []
                # shift_bboxes = []
                for t_mask, bbox in zip(transform_masks, bboxes):
                    pos_mask = deepcopy(t_mask)
                    p = random_click(np.array(pos_mask[0]), point_label, inout)
                    masks+=pos_mask
                    if p is not None:
                        pt.append(p)
                        clear_transform_masks.append(t_mask)
                transform_masks = clear_transform_masks
                pt = np.array(pt)
                point_label = np.ones(pt.shape[:-1])
            else:
                masks = torch.zeros((1, self.img_size, self.img_size))
                no_gt = True

            if no_gt or pt.shape[0]==0:
                return {
                'image': img,
                'label': transform_masks,
                'p_label': torch.tensor([]),
                'pt':torch.tensor([]),
                'bbox':torch.tensor([]),
                'image_meta_dict': image_meta_dict,
                'image_size': [oldw, oldh]
            }
            tmp = deepcopy(pt[...,1])
            pt[...,1] = pt[..., 0]
            pt[...,0] = tmp

            # random the points
            keep = [i for i in range(len(transform_masks))]
            keep_masks = list(zip(transform_masks, keep))
            random.shuffle(keep_masks)
            transform_masks[:], keep[:] = zip(*keep_masks)
            # bboxes = np.array(bboxes)
            categories = np.array(categories)

            return {
                'image': img,
                'label': transform_masks,
                'p_label': point_label[keep],
                'pt': pt[keep],
                'bbox': bboxes[keep],
                'image_meta_dict':image_meta_dict,
                'image_size': [oldw, oldh],
                'category_ids': categories[keep]
            }
    
    def get_mosaic_sample(self, index):
        # loads images in a 4-mosaic
        img4, img_id4, file_name4, boxes4, masks4, categories4 = [], [], [], [], [], []
        s = int(self.img_size / 2)
        # self.mosaic_border = (320, 320)  # min self.img_size * 0.1  # border from edge
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + random.choices(self.dataset_sample_index, k=3)  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            # if i > 0:
            #     break
            img, img_id, file_name, boxes, masks, categories = self.get_sample(index)
            h, w = img.shape[:2]
            # img, _, (h, w) = resize_image( , img)
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                mask_sample = np.zeros_like(img4, np.uint8)[..., 0]
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a: y2a, x1a: x2a] = img[y1b: y2b, x1b: x2b]  # img4[ymin:ymax, xmin:xmax]
            pad_size = [[x1a, x2a, y1a, y2a], [x1b, x2b, y1b, y2b]]
            # Labels
            boxes, masks = shift_coordinates(boxes, masks, pad_size, mask_sample)
            boxes4.append(boxes)
            masks4.append(masks)
            img_id4.append(img_id)
            categories4.append(categories)
            file_name4.append(file_name)

        boxes4 = np.concatenate(boxes4, 0)
        masks4 = np.concatenate(masks4, 0)
        categories4 = np.concatenate(categories4, 0)
        # boxes4, masks4 = clip_coords(img4, boxes4, masks4)
        # vis_mosaic(img4, boxes4, masks4)
        return img4, img_id4, file_name4, boxes4, masks4, categories4
        
    def get_sample(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name'].split('\\')[-1]

        # opencv
        _img = cv2.imread(os.path.join(self.img_dir, path))
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)

        # PIL
        # _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        if self.mode == 'Test':
            return _img, img_metadata['id'], img_metadata['file_name'], None, None, None
        else:
            _masks, _bboxes, _categories = self._gen_seg_mask(
                cocotarget, img_metadata['height'], img_metadata['width'])
            
            if self.mode == 'Training' and len(_masks) != 0:
                augmented = self.aug_transforms(image=_img, masks=_masks, bboxes=_bboxes, category_id=_categories, force_apply=True)
                _img = augmented['image']
                _masks = augmented['masks']
                _keep = list(map(lambda x:x.sum()>100, _masks))
                _masks = np.array(_masks)[_keep]
                _categories = np.array(_categories)[_keep].tolist()
            height, width = _img.shape[:-1]
            if not self.sam_only:
                _bboxes = self._gen_box_prompt(_masks, height, width)
            else:
                _bboxes = np.array(_bboxes)
            return _img, img_metadata['id'], img_metadata['file_name'], _bboxes, _masks, _categories
    
    def _noise_bbox(self, bbox, width, height):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        center_x = (bbox[0] + bbox[2])/2
        center_y = (bbox[1] + bbox[3])/2
        left_random = random.uniform(self.noised_ratio[0], self.noised_ratio[1])
        right_random = random.uniform(self.noised_ratio[0], self.noised_ratio[1])
        top_random = random.uniform(self.noised_ratio[0], self.noised_ratio[1])
        down_random = random.uniform(self.noised_ratio[0], self.noised_ratio[1])
        x1 = max(0, center_x - left_random * w)
        x2 = min(width, center_x + right_random * w)
        y1 = max(0, center_y - top_random * h)
        y2 = min(height, center_y + down_random * h)
        return [x1, y1, x2 , y2]
        
    def _gen_box_prompt(self, masks, height, width):
        bboxes = []
        if self.using_mosaic:
            scale = 1
        elif self.padding and max(height, width) < self.img_size:
            if self.resize_size is not None:
                scale = self.resize_size / max(height, width)
            else:
                scale = 1
        else:
            scale = self.img_size / max(height, width)
        for mask in masks:
            if mask.sum() < 50:
                continue
            pos = np.nonzero(mask)
            x1 = np.min(pos[1]) * scale
            y1 = np.min(pos[0]) * scale
            x2 = np.max(pos[1]) * scale
            y2 = np.max(pos[0]) * scale
            bbox = [x1, y1, x2, y2]
            if self.noised_bbox and self.mode == 'Training':
                bbox = self._noise_bbox(bbox, scale*width, scale*height)
            bboxes.append(bbox)
        return np.array(bboxes)
    
    def _rejust_bbox(self, bbox, height, width):
        if bbox[0] + bbox[2] > width:
            bbox[2] = width - bbox[0]
        if bbox[1] + bbox[3] > height:
            bbox[3] = height - bbox[1]
        return bbox

    def _gen_seg_mask(self, target, height, width):
        masks = []
        bboxes = []
        categories = []
        for instance in target:
            # Fix error of mask conversion
            # img = Image.new('L', (width, height), 0)
            # for polygon in instance['segmentation']:
            #     ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            # mask = np.array(img, dtype=np.uint8)
            if isinstance(instance['segmentation'], list):
                # Polygons
                rle = pymask.merge(pymask.frPyObjects(instance['segmentation'], height, width))
            elif isinstance(instance['segmentation']["counts"], list):
                # Uncompressed RLE
                rle = pymask.frPyObjects(instance['segmentation'], height, width)
            else:
                rle = instance['segmentation']
            
            mask = pymask.decode(rle).astype(np.uint8)
            # mask = np.sum(mask, axis=-1, dtype=np.uint8)
            # mask = Image.fromarray(mask*255)
            bbox = self._rejust_bbox(instance['bbox'], height, width)
            category = instance["category_id"]
            bboxes.append(bbox)
            masks.append(mask)
            categories.append(category)
        return masks, bboxes, categories

    def _preprocess(self, ids):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        return new_ids

    def get_preprocess_shape(self, oldh: int, oldw: int, target_size:int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = target_size / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww)
        newh = int(newh)
        return newh, neww

    def transform_mask(self, masks, height, width, padt, padl, padb, padr):
        composed_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((height, width), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
        masks = composed_transforms(masks)
        masks = F.pad(masks, (padl, padr, padt, padb))
        return masks

    def transform_img(self, image, height, width, padt, padl, padb, padr):
        composed_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])
        normal_transforms = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        image = composed_transforms(image)
        image = F.pad(image, (padl, padr, padt, padb), value=0.447)
        image = normal_transforms(image)
        return image

    def __len__(self):
        return len(self.ids)

def resize_image(img_size, img):
        h0, w0 = img.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

def shift_coordinates(boxes, masks, pad_size, mask_sample):
    x1a, x2a, y1a, y2a = pad_size[0]
    x1b, x2b, y1b, y2b = pad_size[1]
    # pad_w = x1a - x1b
    # pad_h = y1a - y1b
    boxes[:, 0] += x1a
    boxes[:, 1] += y1a
    boxes[:, 2] += x1a
    boxes[:, 3] += y1a
    mosaic_mask = np.zeros((masks.shape[0], mask_sample.shape[0], mask_sample.shape[1]), np.uint8)
    mosaic_mask[:,y1a: y2a, x1a: x2a] = masks[:, y1b: y2b, x1b: x2b]
    return boxes, mosaic_mask

def vis_mosaic(
        img, boxes, masks,
        output_dir="./test_images/mosaic_images",
        img_file="vis.jpg",
):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_path = os.path.join(output_dir, img_file)
    for box in boxes:
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

    color_mask = np.array([0, 0, 255], dtype=np.uint8)
    for mask in masks:
        bbox_mask = mask.astype(bool)
        img[bbox_mask,:] = img[bbox_mask,:] + color_mask * 0.5
    
    cv2.imwrite(img_path, img)
    