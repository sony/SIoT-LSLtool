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
from pathlib import Path

from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType

from lsl_tools.tools.glip_train_net import del_background

class LSL_CVAT(object):
    def __init__(self, CVAT_URL, Username, Password, CVAT_Orgname) -> None:
        self.client = make_client(host=CVAT_URL, credentials=(Username, Password))
        if CVAT_Orgname:
            organizations = [orignization.slug for orignization in self.client.organizations.list()]
            assert CVAT_Orgname in organizations, f"{CVAT_Orgname} is not in CVAT orignaizations names: {organizations}"
            self.client.organization_slug = CVAT_Orgname
        # self.client.api_client.set_default_header("Authorization", f"Token {API_KEY}")
        # self.client.config.status_check_period = 2

    def get_labels(self, coco_json):
        labels = []
        with open(coco_json, "r") as fa:
            categories = json.load(fa)["categories"]
        for category in categories:
            category_id = category["id"]
            label_name = category["name"]
            labels.append({
                "name": label_name,
                "id": category_id,
                "attributes": []
            })
        return labels

    def build_spec(self, coco_json, task_name):
        labels = self.get_labels(coco_json)
        task_spec = {
            "name": f"{task_name}",
            "labels": labels,
        }
        return task_spec

    def build_nobackground(self, coco_json):
        coco_results, _ = del_background(coco_json)
        coco_json = Path(coco_json)
        nobk_coco_json = f"{coco_json.parent}/{coco_json.stem}_nobackground.json"
        with open(nobk_coco_json, "w+") as  fb:
            json.dump(coco_results, fb)
        return nobk_coco_json

    def create_task(self, img_dir, coco_json, task_name):
        
        # The current export function of cvat does not support uploading category_id starting from 0.
        coco_json = self.build_nobackground(coco_json)

        task_spec = self.build_spec(coco_json, task_name)

        img_files = os.listdir(img_dir)
        resource_list = [f"{img_dir}/{img_file}" for img_file in img_files]

        task = self.client.tasks.create_from_data(
            spec=task_spec,
            resource_type=ResourceType.LOCAL,
            resources=resource_list,            
        )
        task.import_annotations("COCO 1.0", coco_json, pbar=None)
        # If an object is modified on the server, the local object is not updated automatically.
        # To reflect the latest changes, the local object needs to be fetch()-ed.
        task.fetch()
    

    def export_to_cvat(self, img_dir, coco_json, task_name):
        self.create_task(img_dir, coco_json, task_name)
