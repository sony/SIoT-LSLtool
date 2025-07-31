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
import yaml
import os
import os.path as opt
from pathlib import Path

from lsl_tools.cvat.lsl_inference.utils import get_categories


def build_spec(categories):
    num_classes = len(categories)
    spec = "["
    for idx, category in enumerate(categories):
        spec += '{"id": '
        spec += str(category["id"])
        spec += ', "name": "'
        spec += str(category["name"])
        spec += '"}'
        if idx < num_classes - 1:
            spec += ', ' 
    spec+="]"
    return spec

def update_yaml(yaml_file, task, project_dir, base_docker_image):

    project_dir = Path(project_dir)
    serverless_name = f"lsl_{task}_{project_dir.parent.parent.stem}_{project_dir.stem}"
    
    # update metadata.name
    yaml_file["metadata"]["name"] = serverless_name
    yaml_file["metadata"]["annotations"]["name"] = serverless_name
    
    # update spec.build.baseImage
    yaml_file["spec"]["build"]["baseImage"] = base_docker_image

    # update spec.description
    yaml_file["spec"]["description"] = serverless_name

    # update spec.build.image
    yaml_file["spec"]["build"]["image"] = f"{serverless_name}:latest"
    return yaml_file

def update_anno_spec(final_yaml_path, project_dir):
    categories = get_categories(project_dir)
    spec = build_spec(categories)
    with open(final_yaml_path, "r") as fa:
        yaml_lines = fa.readlines()
    for idx, yaml_line in enumerate(yaml_lines):
        if yaml_line == '    spec: \'[{ "id": 1, "name": "class_name" }]\n':
            # yaml_lines[idx] = f"    spec: | \n      {spec}\n"
            yaml_lines[idx+2] = ""
            yaml_lines[idx+1] = ""
            yaml_lines[idx] = f'    spec: \'{spec}\'\n'

    with open(final_yaml_path, "w") as fs:
        fs.writelines(yaml_lines)


def build_yaml(project_dir, task, base_docker_image):
    file_dir = opt.dirname(__file__)

    severless_dir = opt.join(file_dir, f"{task}")
    base_yaml_path = f"{severless_dir}/function_base.yaml"
    final_yaml_path = f"{severless_dir}/function.yaml"
    with open(base_yaml_path, "r") as f:
        yaml_file = yaml.load(f, Loader=yaml.Loader)
    yaml_file = update_yaml(yaml_file, task, project_dir, base_docker_image)
    with open(final_yaml_path, "w") as fb:
        yaml.dump(yaml_file, fb, default_flow_style=False)
    update_anno_spec(final_yaml_path, project_dir)


if __name__ == "__main__":
    build_yaml("${lsl_project}", "fsis", "zerorpc_demo:v1")