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
from tqdm import trange
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms


class COCO2ClassDataset(Dataset):
    def __init__(
        self, args, ann_file, img_dir, category2index, mode = 'Training'
    ):
        self.mode = mode
        self.img_size = args.image_size
        self.img_dir = img_dir
        self.category2index = category2index
        if mode == "Test":
            self.coco=None
            self.img_files = os.listdir(img_dir)
            self.ids = [i for i in range(len(self.img_files))]
        else:
            self.coco = COCO(ann_file)
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids)
        
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([args.image_size, args.image_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.valid_transform = transforms.Compose([ 
                transforms.ToPILImage(),
                transforms.Resize([args.image_size, args.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        img, label, image_id, file_name, width, height = self.get_sample(index)
        if self.mode == "Test":
            return img, image_id, file_name, width, height
        else:
            return img, label, image_id, file_name
    
    def get_sample(self, index):
        coco = self.coco
        img_id = self.ids[index]
        if coco is not None:
            img_metadata = coco.loadImgs(img_id)[0]
            file_name = img_metadata['file_name'].split('\\')[-1]
            coco_ann = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            if len(coco_ann)>=1:
                coco_ann = coco_ann[0]
                category_id = coco_ann["category_id"]
                label = self.category2index[category_id]
            else:
                label = 0
            
        else:
            file_name = self.img_files[img_id]
            label = None

        _img = cv2.imread(os.path.join(self.img_dir, file_name))
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        ori_height, ori_width = _img.shape[:2]
        if self.mode == "Training":
            _img = self.train_transform(_img)
        else:
             _img = self.valid_transform(_img)

        return _img, label, img_id, file_name, ori_width, ori_height

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
    
    def __len__(self):
        return len(self.ids)

    