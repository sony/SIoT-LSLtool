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
from label_studio_sdk import Client


class LSL_LabelStudio(object):
    def __init__(self, LABEL_STUDIO_URL, API_KEY) -> None:
        self.ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
        self.ls.check_connection()

    def create_project(self, labelstudio_xml, project_name):
        with open(labelstudio_xml, "r") as f:
            label_config = f.read()
        self.project = self.ls.start_project(
            title=project_name,
            label_config=label_config
            )

    def import_images(self, img_dir):
        img_files = os.listdir(img_dir)
        for img_file in img_files:
            if img_file.endswith(".jpg") or img_file.endswith(".png") or img_file.endswith(".JPEG"):
                fname = f"{img_dir}/{img_file}"
                self.project.import_tasks(fname)

    def taskid_filename_dict(self, tasks):
        filename2taskid = {}
        for task in tasks:
            filename = os.path.basename(task["data"]["image"])
            prefix = filename.split("-")[0] + "-"
            filename = filename.replace(prefix, "")
            filename2taskid[filename] = task["id"]
        return filename2taskid


    def add_predictions(self, labelstudio_json):
        filename2taskid = self.taskid_filename_dict(self.project.tasks)
        with open(labelstudio_json, "r") as fb:
            predictions_bk = json.load(fb)
        for img_prediction in predictions_bk:
            file_name = os.path.basename(img_prediction["data"]["image"])
            pred_label = img_prediction["annotations"][0]["result"]
            task_id = filename2taskid[file_name]
   
            self.project.create_prediction(task_id, pred_label)
        print(f'tasks have been preannotated with lsl predictions')
    
    def export_labelstudio(self, labelstudio_json, labelstudio_xml, project_name, img_dir):
        self.create_project(labelstudio_xml, project_name)
        self.import_images(img_dir)
        self.add_predictions(labelstudio_json)


if __name__=="__main__":
    labelstudio_json = "/home/zhanglei/work/project/dataset_tools/wheats/hok_wheat/lab/wheat.json"
    labelstudio_xml = "/home/zhanglei/work/project/dataset_tools/wheats/hok_wheat/lab/wheat.label_config.xml"
    img_dir = "/home/zhanglei/work/project/dataset_tools/wheats/hok_wheat/valid"
    LABEL_STUDIO_URL = 'http://localhost:8080'
    API_KEY = '88184212bba02066eb6cbf4a1baad388c9f22c33'
    project_name = "hok_wheat_lsl_test"
    labestudio_ci = LSL_LabelStudio(LABEL_STUDIO_URL, API_KEY)
    labestudio_ci.export_labelstudio(labelstudio_json, labelstudio_xml, project_name, img_dir)





