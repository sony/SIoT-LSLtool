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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from tqdm import tqdm

from ..utils.utils import *
from ..utils.set_random import set_random_seed
from ..data.coco2class import COCO2ClassDataset
from ..models.get_network import get_network

class Mobilenetv2Training():
    def __init__(self):
        self.best_acc = 0.0
        self.best_tol = 1e8
        self.smallest_loss = 1e8
        self.best_hyperparameters = {}
        self.require_train = True

    def prepare_dataloader(self, args, valid_mode="Valid"):
        train_img_dir = args.train
        train_ann_file = args.train_ann
        val_img_dir = args.val
        val_ann_file = args.val_ann
        if args.data == 'coco':
            if valid_mode == "Test":
                train_loader =None
            else:
                coco_train_dataset = COCO2ClassDataset(args, train_ann_file, train_img_dir, self.category2index, mode = 'Training')
                train_loader = DataLoader(coco_train_dataset, batch_size=args.ims_per_batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            coco_val_dataset = COCO2ClassDataset(args, val_ann_file, val_img_dir, self.category2index, mode = valid_mode)
            val_loader = DataLoader(coco_val_dataset, batch_size=args.ims_per_batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return train_loader, val_loader

    def train_epoch(self, args, train_loader, epoch):
        self.model.train()
        GPUdevice = torch.device('cuda:' + str(args.gpu_device))
        epoch_loss = 0
        ind = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
            for image, label, _, _ in train_loader:
                ind+=1
                self.optimizer.zero_grad()
                image = image.to(device = GPUdevice)
                label = label.to(device = GPUdevice)
                pred = self.model(image)
                loss = self.loss_func(pred, label)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                pbar.update()
        self.scheduler.step()  

    def load_checkpoint(self, args, check=False):
        print(f'=> resuming from {self.checkpoint_path}')
        if os.path.exists(self.checkpoint_path):
            if check:
                while True:
                    answer = input("The pre-trained MobileNetV2 model already exists, Whether to retrain it, enter Y or N: ")
                    if answer.lower() == "n":
                        loc = 'cuda:{}'.format(args.gpu_device)
                        checkpoint = torch.load(self.checkpoint_path, map_location=loc)
                        self.model.load_state_dict(checkpoint)
                        print(f'=> loaded checkpoint {self.checkpoint_path}')
                        self.require_train = False
                        break
                    elif answer.lower() == "y":
                        break
                    else:
                        print("Input error, please enter Y or N")
            else:
                loc = 'cuda:{}'.format(args.gpu_device)
                checkpoint = torch.load(self.checkpoint_path, map_location=loc)
                self.model.load_state_dict(checkpoint)
                print(f'=> loaded checkpoint {self.checkpoint_path}')
                self.require_train = False
        else:
            dir_name = os.path.dirname(self.checkpoint_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

    def save_checkpoint(self, args, tol):
        if args.distributed:
            checkpoint = self.model.module.state_dict()
        else:
            checkpoint = self.model.state_dict()
        if tol <= self.best_tol:
            self.best_tol = tol
            is_best = True
            if is_best:
                torch.save(checkpoint, self.checkpoint_path)
            print(f"Saving the best model in {self.checkpoint_path}")
    
    def prepare_cate_info(self, project_dir):
        if not os.path.exists(f"{project_dir}/cate_info.json"):
            with open(f"{project_dir}/cate_info.json", "w+") as fa:
                json.dump(self.categories, fa)

    def preapre(self, args, check=False):
        gpu_device = torch.device('cuda', args.gpu_device)

        model = get_network(args, "mobilenetv2", use_gpu=args.gpu, gpu_device=gpu_device, distribution = args.distributed)
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.step, gamma=args.gamma) #learning rate decay

        self.checkpoint_path = os.path.join(args.output_dir, args.exp_name, "Model", "mb2_checkpoint_best.pth")
        self.model = model
        self.load_checkpoint(args, check)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = nn.CrossEntropyLoss()
        
        val_coco = COCO(args.val_ann)
        self.categories = val_coco.loadCats(val_coco.getCatIds())
        self.category2index = {category["id"]:idx+1 for idx, category in enumerate(self.categories)}
        self.index2category = {idx+1:category["id"] for idx, category in enumerate(self.categories)}
       
    def train(self, args, project_dir):
        set_random_seed(seed=0)
        self.preapre(args, check=True)
        self.prepare_cate_info(project_dir=project_dir)
        train_loader, val_loader = self.prepare_dataloader(args)
        if self.require_train:
            for epoch in range(args.max_epochs):
                self.train_epoch(args, train_loader, epoch)
                if epoch % args.val_step == 0 or epoch == args.max_epochs-1:
                    torch.cuda.empty_cache()
                    self.valid(args, val_loader)
                torch.cuda.empty_cache()
        # after training do test
        self.valid_best_checkpoint(args)
    
    def valid(self, args, val_loader):
        GPUdevice = torch.device('cuda:' + str(args.gpu_device))
        self.model.eval()
        total_correct = 0
        total_size = 0
        for images, labels, _, _ in tqdm(val_loader):
            images = images.to(device = GPUdevice)
            labels = labels.to(device = GPUdevice)
            with torch.no_grad():
                outputs = self.model(images)
                _, pred_index = torch.max(outputs, dim=1)
            total_size += labels.size(0)
            total_correct += (pred_index == labels).sum().item()
        
        accuracy = total_correct / total_size
        print('Accuracy of the network: %d %%' % (100 * accuracy))

        self.save_checkpoint(args, accuracy)

    def inference_pseudo_label(self, args, val_loader):
        GPUdevice = torch.device('cuda:' + str(args.gpu_device))
        self.model.eval()
        anns_list = []
        img_list = []

        with open(args.val_ann, "r") as fa:
            coco_results = json.load(fa) 
        ind = 0
        for images, image_ids, file_names, widths, heights in tqdm(val_loader):
            images = images.to(device = GPUdevice)
            with torch.no_grad():
                outputs = self.model(images)
                _, pred_indexs = torch.max(outputs, dim=1)
            for idx in range(len(image_ids)):
                ind+=1
                image_id = image_ids[idx].item()
                pred_idx = pred_indexs[idx].item()
                if pred_idx == 0:
                    continue
                category_id = self.index2category[pred_idx]
                width = widths[idx].item()
                height = heights[idx].item()
                img_list.append(
                    {
                    "id": image_id,
                    "file_name": file_names[idx],
                    "height": height,
                    "width": width
                    }
                )
                anns_list.append(
                    {
                    "image_id": image_id,
                    "iscrowd": 0, 
                    "id": ind, 
                    "area": width*height, 
                    "score": 1, 
                    "bbox": [0, 0, width, height], 
                    "segmentation": [], 
                    "category_id": category_id
                    })
        
        coco_results["images"] = img_list
        coco_results["annotations"] = anns_list
        return coco_results

    def valid_best_checkpoint(self, args):
        self.preapre(args)
        _, val_loader = self.prepare_dataloader(args)
        self.model.eval()
        # val_img_dir = os.path.join(args.data_root, args.val)
        self.valid(args, val_loader)
    
    def inference_label(self, args, img_dir, result_path):
        self.preapre(args)
        args.val = img_dir
        # args.val_ann = None
        _, val_loader = self.prepare_dataloader(args, valid_mode="Test")
        self.model.eval()
        anns = self.inference_pseudo_label(args, val_loader)

        with open(result_path, "w") as fb:
            json.dump(anns, fb)
