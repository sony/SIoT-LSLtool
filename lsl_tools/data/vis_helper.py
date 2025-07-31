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
from shutil import copyfile


def copy_img_file(root, split, save_to="images/train"):
    file_path = os.path.join(root, split)
    with open(file_path, 'r') as fo:
        file_ls = [file.strip() for file in fo.readlines()]
        file_ls = [file.split(" ")[0] for file in file_ls if int(file.split(" ")[-1]) > 0]

    os.makedirs(save_to, exist_ok=True)
    for file in file_ls:
        copyfile(f"{root}/JPEGImages/{file}.jpg", f"{save_to}/{file}.jpg")


def copy_ann_file(root, split, save_to="annotations/train"):
    file_path = os.path.join(root, split)
    with open(file_path, 'r') as fo:
        file_ls = [file.strip() for file in fo.readlines()]

    os.makedirs(save_to, exist_ok=True)
    for file in file_ls:
        copyfile(f"{root}/Annotations/{file}.xml", f"{save_to}/{file}.xml")


def voc2coco_visual_check(
    json_file="datasets/bus_car/annotations/train.json",
    img_dir="datasets/bus_car/train",
    random_show_num=20,
    show_img_id=None,
    show_ori=False,
):
    """
    visual check func to valid voc2coco transformation

    Args:
        json_file (str): path/to/coco/ann_file
        img_dir (srt): path/to/coco/images
        random_show_num (int): random selected number of images to visualize
        show_img_id (int): if specified, the func will show that image
        show_ori (bool): whether to see the coupled origin images of the visualized one
    Returns:
        plt images
    """
    from pycocotools.coco import COCO
    import random
    from PIL import Image
    import matplotlib.pyplot as plt

    coco = COCO(json_file)
    cats = coco.loadCats(coco.getCatIds())  # catNms=['car', 'bus'] which can be used as filter
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    catIds = coco.getCatIds()  # can be used as filters
    imgIds = coco.getImgIds()

    def img_ann_show(img_id):
        img = coco.loadImgs(img_id)[0]
        I = Image.open(os.path.join(img_dir, img["file_name"]))
        if show_ori:
            plt.axis('off')
            plt.imshow(I)
            plt.show()

        plt.imshow(I)
        plt.axis('off')
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns, draw_bbox=True)
        plt.show()

    if show_img_id:
        img_ann_show(show_img_id)
        return

    random_show_num = min(random_show_num, len(imgIds))
    sampled_ids = random.sample(imgIds, random_show_num)
    for idx in sampled_ids:  # imgIds[np.random.randint(0, len(imgIds))]
        img_ann_show(idx)


if __name__ == "__main__":
    os.chdir("/home/kevin/22-lsl_tools")

    # copy_img_file(
    #     root="datasets/VOC2012",
    #     split="ImageSets/Main/car_trainval.txt",
    #     save_to="datasets/bus_car/weak_supervise_full/car"
    # )

    # copy_ann_file(
    #     root="datasets/VOC2012",
    #     split="ImageSets/Main/voc12_bus_car_trainval.txt",
    #     save_to="datasets/bus_car/annotations/train",
    # )

    voc2coco_visual_check()