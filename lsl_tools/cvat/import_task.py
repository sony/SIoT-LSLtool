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
import numpy as np
from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType


class Import_CVAT(object):
    def __init__(self, CVAT_URL, Username, Password) -> None:
        self.CVAT_URL = CVAT_URL
        # self.API_KEY = API_KEY
        self.client = make_client(host=CVAT_URL, credentials=(Username, Password))
        # self.client.api_client.set_default_header("Authorization", f"Token {API_KEY}")
        # self.client.config.status_check_period = 2

    def load_task(self, task_id):
        self.task = self.client.tasks.retrieve(task_id)

    def unzip_dataset(self, output_file, task_name):
        output_dir = os.path.dirname(output_file)
        unzip_cmd = f"unzip -d {output_dir} {output_file}"
        assert os.system(unzip_cmd) == 0, f"Fail to unzip {output_file}"
        img_dir = f"{output_dir}/images"
        annotation_path = f"{output_dir}/annotations/{task_name}_default.json"
        with open(annotation_path, "r") as f:
            annotations = json.load(f)
        num_imgs = len(annotations["images"])
        return img_dir, num_imgs, annotation_path

    def export_task(self, task_id, output_root, task="fsis"):
        # if task  == "fsis":
        #     task_name = "instances"
        # else:
        #     task_name = "bboxes"
        task_name = "instances"

        self.load_task(task_id)
        zip_filename = f"{output_root}/tmp.zip"
        self.task.export_dataset(format_name = "COCO 1.0", filename=zip_filename)
        return self.unzip_dataset(zip_filename, task_name)

if __name__=="__main__":
    task_id = 21
    CVAT_URL = 'http://xxxx:8080'
    API_KEY = 'xxx'
    output_root = "xxx"
    labestudio_ci = Import_CVAT(CVAT_URL, API_KEY)
    labestudio_ci.export_task(task_id, output_root)





