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
import shutil

from lsl_tools.tools import lsl_args


class SlideWindow(object):
    def __init__(self, height, width, overlap, object_threshold):
        self.height = height
        self.width = width
        self.overlap = overlap
        self.object_threshold = object_threshold

    def crop(self, img_dir, output_dir, ann_path=None):
        slidewindow_script = os.path.normpath(f'{lsl_args.package_top}/data/slide_windows.py')
        slidewindow_args = f' --height {self.height}'
        slidewindow_args += f' --width {self.width}'
        slidewindow_args += f' --overlap {self.overlap}'
        slidewindow_args += f' --percentage {self.object_threshold}'
        slidewindow_args += f' --img-dir {img_dir}'
        slidewindow_args += f' --output-dir {output_dir}'
        if ann_path:
            slidewindow_args += f' --ann-path {ann_path}'
        slidewindow_cmd = f'{sys.executable} {slidewindow_script} {slidewindow_args}'
        assert os.system(slidewindow_cmd) == 0, f'Fail to do Slidewindows on images in the {img_dir}'

    def restore(self, img_dir, pred_ann_path, output_dir, score_threshold, semantic=False):
        slidewindow_args = f' --height {self.height}'
        slidewindow_args += f' --width {self.width}'
        slidewindow_args += f' --overlap {self.overlap}'
        slidewindow_args += f' --img-dir {img_dir}'
        slidewindow_args += f' --pred-ann-path {pred_ann_path}'
        slidewindow_args += f' --output-ann-path {output_dir}'
        slidewindow_args += f' --score-threshold {score_threshold}'
        if semantic:
            slidewindow_args += f' --semantic True'
        slidewindow_script = os.path.normpath(f'{lsl_args.package_top}/data/slide_windows_restore.py')
        slidewindow_cmd = f'{sys.executable} {slidewindow_script} {slidewindow_args}'
        assert os.system(slidewindow_cmd) == 0, f'Fail to use slidewindow_restore to merge the cropped images in the {img_dir}'

    def run(self, img_dir, output_root, cropped_name, ann_path=None):
        cropped_ann_path = None
        output_dir = os.path.join(output_root, cropped_name)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        self.crop(img_dir, output_dir, ann_path)
        if ann_path:
            cropped_ann_path = f"{output_dir}/annotations/{os.path.basename(ann_path)}"
        cropped_img_dir = f"{output_dir}/{os.path.basename(img_dir)}"
        num_cropped_imgs = len(os.listdir(cropped_img_dir))
        return [cropped_img_dir, cropped_ann_path, num_cropped_imgs]
        
    def prepare_dataset(self, data_info, train_names, valid_name, test_name=None):
        cropped_test_name = None
        output_root = os.path.join(lsl_args.voc_tmp, "slidewindows_data")
        if not os.path.exists(output_root):
            os.mkdir(output_root)

        train = data_info["labeled"][train_names[0]]
        cropped_source_name = f"{train_names[0]}_{self.width}_{self.height}_{self.overlap}"
        cropped_train_infos = self.run(img_dir=train["img"], output_root=output_root, cropped_name=cropped_source_name, ann_path=train["ann"])
        data_info["labeled"][train_names[0]] = {"ann":cropped_train_infos[1], "fmt": "coco", "img": cropped_train_infos[0], "num": cropped_train_infos[2]}


        valid = data_info["labeled"][valid_name]
        cropped_valid_name = f"{valid_name}_{self.width}_{self.height}_{self.overlap}"
        cropped_valid_infos = self.run(img_dir=valid["img"], output_root=output_root, cropped_name=cropped_valid_name, ann_path=valid["ann"])
        data_info["labeled"][valid_name] = {"ann":cropped_valid_infos[1], "fmt": "coco", "img": cropped_valid_infos[0], "num": cropped_valid_infos[2]}

        if test_name:
            target = data_info["unlabeled"][test_name]
            cropped_test_name = f"{test_name}_{self.width}_{self.height}_{self.overlap}"
            cropped_target_infos = self.run(img_dir=target["img"], output_root=output_root, cropped_name=cropped_test_name)
            data_info["unlabeled"][cropped_test_name] = {"fmt": "coco", "auto_label":None, "img": cropped_target_infos[0], "num": cropped_target_infos[2]}
        
        return data_info, cropped_test_name
    
def get_slidewindow(label_cmds):

    width, height=label_cmds.slidewindow_size
    overlap=label_cmds.overlap
    object_threshold=label_cmds.object_threshold
    return SlideWindow(height, width, overlap, object_threshold)



