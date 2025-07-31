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
import torch.optim as optim
from torch.cuda.amp import GradScaler
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from ..engine.train_sam import train_sam, valid_sam
from ..engine.predictor import predict
from ..data.coco_mosaic import COCODataset
from ..models.get_network import get_network
from ..utils.set_random import set_random_seed
from ..utils.save_checkpoints import save_checkpoint
from ..utils.set_log import set_path_dir


class Training():
    def __init__(self):
        self.best_acc = 0.0
        self.best_tol = 1e8
        self.smallest_loss = 1e8
        self.best_hyperparameters = {}

    def prepare_dataloader(self, args, test=False):
        train_img_dir = args.train
        train_ann_file = args.train_ann
        val_img_dir = args.val
        val_ann_file = args.val_ann
        if test:
            valid_mode = "Test"
        else:
            valid_mode = "Valid"

        if args.data == 'coco':
            if valid_mode == "Test":
                train_loader =None
            else:
                coco_train_dataset = COCODataset(args, train_ann_file, train_img_dir, mode = 'Training')
                train_loader = DataLoader(coco_train_dataset, batch_size=args.ims_per_batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            coco_val_dataset = COCODataset(args, val_ann_file, val_img_dir, mode = valid_mode)
            val_loader = DataLoader(coco_val_dataset, batch_size=args.ims_per_batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        with open(val_ann_file, "r") as f:
            GT_coco = json.load(f)
        img2id = {}
        for img_info in GT_coco["images"]:
            img2id[img_info["file_name"]] = [img_info["id"], int(img_info["width"]), int(img_info["height"])]
        return train_loader, val_loader, img2id

    def train_epoch_sam(self, args, train_loader, epoch, scaler, additional=''):
        self.model.train()
        loss, dice_loss, ce_loss = train_sam(args, self.model, self.optimizer, self.scheduler, train_loader, epoch, scaler)
        print(f'Train loss: {loss}|| @ epoch {epoch}|| @ lr {self.optimizer.param_groups[0]["lr"]}.')

    def load_checkpoint(self, args, model, optimizer):
        print(f'=> resuming from {args.weight}')
        assert os.path.exists(args.weight)
        checkpoint_file = os.path.join(args.weight)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)

        if 'state_dict' not in checkpoint.keys():  # offical sam model
            start_epoch = 0
            model.load_state_dict(checkpoint)
        else:
            self.best_tol = checkpoint['best_tol']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.path_infos = checkpoint['path_helper']
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        return model, optimizer, start_epoch
    
    def save_checkpoint(self, args, tol, epoch):
        if args.distributed:
            sd = self.model.module.state_dict()
        else:
            sd = self.model.state_dict()
        if tol <= self.best_tol:
            self.best_tol = tol
            is_best = True

            save_checkpoint({
            'epoch': epoch + 1,
            'model': self.model,
            'state_dict': sd,
            'optimizer': self.optimizer.state_dict(),
            'best_tol': self.best_tol,
            'path_helper': args.path_infos,
            }, is_best, args.path_infos['ckpt_path'], filename="best_checkpoint")
            print(f"Saving the best model in {args.path_infos['ckpt_path']}")

    def preapre(self, args):
        start_epoch = 0
        gpu_device = torch.device('cuda', args.gpu_device)

        model = get_network(args, args.net, use_gpu=args.gpu, gpu_device=gpu_device, distribution = args.distributed)
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        if args.weight:
            model, optimizer, start_epoch = self.load_checkpoint(args, model, optimizer)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.step, gamma=args.gamma) #learning rate decay
        args.path_infos = set_path_dir(args.output_dir, args.exp_name)
        set_path_dir(args.output_dir, args.exp_name)
        # print(args)
        # checkpoint_path = os.path.join('checkpoint', args.net, TIME_NOW)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        return start_epoch
   
        
    def train(self, args):
        set_random_seed(seed=0)
        scaler = GradScaler()
        start_epoch = self.preapre(args)
        train_loader, val_loader, img2id = self.prepare_dataloader(args)
        for epoch in range(start_epoch, args.max_epochs):
            if args.net in ['sam', "sam-l", 'sam-adapter']:
                self.train_epoch_sam(args, train_loader, epoch, scaler)
            if epoch % args.val_step == 0 or epoch == args.max_epochs-1:
                torch.cuda.empty_cache()
                self.valid(args, val_loader, epoch, img2id)
            torch.cuda.empty_cache()
        # after training do test
        args.weight = os.path.join(args.path_infos["ckpt_path"], "checkpoint_best.pth")

        self.valid_best_checkpoint(args)
    
    def valid(self, args, val_loader, epoch, img2id):
        self.model.eval()
        if args.net in ["sam", "sam-l", "sam-adapter"]:
            if not args.sam_only:
                # GT_anns, loss, dice_loss, ce_loss = valid_sam(args, self.model, val_loader, img2id)
                val_ann_dir = os.path.join(args.data_root, args.val_ann)
                cocoGT = COCO(val_ann_dir)

                # cocoDT = cocoGT.loadRes(GT_anns)
                # cocoEval_GT = COCOeval(cocoGT, cocoDT, "segm")
                # cocoEval_GT.evaluate()
                # cocoEval_GT.accumulate()
                # cocoEval_GT.summarize()
            
                GLIP_anns = predict(args, val_loader, self.model, img2id)
                cocoGLIP = cocoGT.loadRes(GLIP_anns)
                cocoEval_GLIP = COCOeval(cocoGT, cocoGLIP, "segm")
                cocoEval_GLIP.evaluate()
                cocoEval_GLIP.accumulate()
                cocoEval_GLIP.summarize()

                tol = -cocoEval_GLIP.stats[0]

            else:
                GLIP_anns, mean_iou = predict(args, val_loader, self.model, img2id)
                tol = - mean_iou
                print(f"Mean IoU: {mean_iou}")                
        
        self.save_checkpoint(args, tol, epoch)

    def valid_best_checkpoint(self, args):
        start_epoch = self.preapre(args)
        _, val_loader, img2id = self.prepare_dataloader(args)
        self.model.eval()
        # val_img_dir = os.path.join(args.data_root, args.val)
        val_ann_dir = os.path.join(args.data_root, args.val_ann)
        if not args.sam_only:
            anns = predict(args, val_loader, self.model, img2id)
        else:
            anns, mean_iou = predict(args, val_loader, self.model, img2id)
        with open(f"{args.path_infos['prefix']}/predictions.json", "w") as fb:
            json.dump(anns, fb)
        print(f"Saving the coco format result in {args.path_infos['prefix']}/predictions.json")

        # evalutate
        if not args.sam_only:
            annTypes = ["bbox", "segm"]
            cocoGT = COCO(val_ann_dir)
            cocoDT=cocoGT.loadRes(anns)
            # for annType in annTypes:
            box_cocoEval = COCOeval(cocoGT,cocoDT,annTypes[0])
            box_cocoEval.evaluate()
            box_cocoEval.accumulate()
            box_cocoEval.summarize()

            mask_cocoEval = COCOeval(cocoGT,cocoDT,annTypes[1])
            mask_cocoEval.evaluate()
            mask_cocoEval.accumulate()
            mask_cocoEval.summarize()
        else:
            print(f"Mean IoU: {mean_iou}")
    
    def valid_base_sam(self, args):
        args.weight = None
        _ = self.preapre(args)
        _, val_loader, img2id = self.prepare_dataloader(args)
        self.model.eval()
        # val_img_dir = os.path.join(args.data_root, args.val)
        val_ann_dir = os.path.join(args.data_root, args.val_ann)
        anns = predict(args, val_loader, self.model, img2id)
        with open(f"{args.path_infos['prefix']}/predictions.json", "w") as fb:
            json.dump(anns, fb)
        print(f"Saving the coco format result in {args.path_infos['prefix']}/predictions.json")

        # evalutate
        annTypes = ["bbox", "segm"]
        cocoGT = COCO(val_ann_dir)
        cocoDT=cocoGT.loadRes(anns)
        # for annType in annTypes:
        box_cocoEval = COCOeval(cocoGT,cocoDT,annTypes[0])
        box_cocoEval.evaluate()
        box_cocoEval.accumulate()
        box_cocoEval.summarize()

        mask_cocoEval = COCOeval(cocoGT,cocoDT,annTypes[1])
        mask_cocoEval.evaluate()
        mask_cocoEval.accumulate()
        mask_cocoEval.summarize()
    
    def inference_labels(self, args, img_dir, prompt_path, result_path, conf_thresh, no_train):
        start_epoch = self.preapre(args)
        args.weight = None
        if not no_train:
            args.weight = os.path.join(args.path_infos["ckpt_path"], "checkpoint_best.pth")  
          
        args.val = img_dir
        args.val_ann = prompt_path
        args.prompt = prompt_path
        _, val_loader, img2id = self.prepare_dataloader(args, test=True)
        self.model.eval()
        anns = predict(args, val_loader, self.model, img2id, mode="Test")
        anns = list(filter(lambda ann: ann["score"] >= conf_thresh, anns))
        with open(args.prompt, "r") as fs:
            result = json.load(fs)
        result["annotations"] = anns
        with open(result_path, "w") as fb:
            json.dump(result, fb)