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
import yaml
from tabulate import tabulate
from lsl_tools.cli.utils.config_dict import CfgDict


class DataManager:
    """ Maintain project data info while in lsl process

    the data_source structure will be:
        "data_source" : {
            "labeled": {
                "name1": {"ann": "ann_data_path1", "img": "img_data_path1", },
                "name2": {"ann": "ann_data_path2", "img": "img_data_path2", },
                 ...,
            },
            "unlabeled": {
                "name3": {"img": "img_data_path3", },
                "name4": {"img": "img_data_path4", },
                ...,
            },
        }
    the parameters should be specified in
    `lsl import --annotation anno_data_path --image image_path --name name [--format fmt]`
    `lsl import --unlabeled-image img_path --name name [--format fmt]`
    Note that "ann" mean to annotation throughout the program

    The idea is that when in process the attr:config_state record the entire info of current
    project and attr:registerer_name, attr:_labeled_src, attr:_unlabeled_src record temp data
    exchanging info. More fields can be added for extension.
    """
    def __init__(self, cfg_path):

        self.config_path = cfg_path
        self.config_state = None
        self.registered_name = []
        self._labeled_src = {}
        self._unlabeled_src = {}

    def update(self):
        """restore last operation state and information"""
        with open(self.config_path, 'r') as fo:
            def forward_slash(d):
                for k,v in d.items():
                    if isinstance(v, dict):
                        forward_slash(v)
                    else:
                        if k.lower() in ['img', 'ann', 'init'] and d[k] is not None:
                            d[k] = v.replace(os.sep, '/')
            config = yaml.full_load(fo)
            # fix 'unicodeescape' issue on Windows
            forward_slash(config)
        self.config_state = CfgDict(config)
        data_source = self.config_state.data_source
        if "labeled" in data_source.keys():
            self._labeled_src.update(data_source.labeled)
        if "unlabeled" in data_source.keys():
            self._unlabeled_src.update(data_source.unlabeled)

        self.registered_name = self.config_state.registered

    def record_data_src(self):
        """ write current config_state to config file """

        # update data_source info when writing
        self.config_state.data_source = self.get_data_src()
        # config = CfgDict(self.config_state)
        config = eval(self.config_state.__repr__())
        with open(self.config_path, 'w') as fo:
            yaml.dump(config, fo)

    def get_data_src(self, name=None):
        """
        default return whole data source info optionally specified data source by name
        """
        if not name:
            return {
                "labeled": self._labeled_src,
                "unlabeled": self._unlabeled_src,
            }

        if self._labeled_src.get(name, None):
            return self._labeled_src[name]
        else:
            return self._unlabeled_src[name]

    def get_register_list(self):
        return self.registered_name

    def add_data_src(self, source_info, labeled, name=0):
        if name in self.registered_name:
            print("The name has been registered, please specify another name")
            exit(0)
        else:
            self.registered_name.append(name)

        if labeled:
            self._labeled_src[name] = source_info
        else:
            self._unlabeled_src[name] = source_info

    def add_training_info(self, data_into, name, record=True):
        self._unlabeled_src[name].update(data_into)
        if record:
            self.record_data_src()

    def remove_data_src(self, name):

        l_names = self._labeled_src.keys()
        u_names = self._unlabeled_src.keys()

        if name in l_names:
            self._labeled_src.pop(name)
            print(f"Data source : labeled | {name} has been removed")
        elif name in u_names:
            self._unlabeled_src.pop(name)
            print(f"Data source : unlabeled | {name} has been removed")
        else:
            print(f"Data source named {name} doesn't exist")
        self.registered_name.remove(name)

        self.record_data_src()

    def rename_data_src(self, old_name, new_name):
        l_names = self._labeled_src.keys()
        u_names = self._unlabeled_src.keys()

        if old_name in l_names:
            self._labeled_src[new_name] = self._labeled_src.pop(old_name)
            print(f"Data source : labeled | {old_name} has been renamed to {new_name}")
        elif old_name in u_names:
            self._unlabeled_src[new_name] = self._unlabeled_src.pop(old_name)
            print(f"Data source : unlabeled | {old_name} has been renamed to {new_name}")
        else:
            print(f"Data source named {old_name} doesn't exist")

        self.registered_name.remove(old_name)
        self.registered_name.append(new_name)

        self.record_data_src()

    def show_data_info(self):
        info = self.config_state.data_source
        info_dict = eval(info.__repr__())
        table_header = ["set", "source"]
        exp_table = [
            (str(k), yaml.dump(v, default_flow_style=False))
            for k, v in info_dict.items()
        ]
        print(tabulate(exp_table, headers=table_header, tablefmt="fancy_grid"))

    def get_data_path(self, file_names):
        return [self._unlabeled_src[name].img for name in file_names]
    
    def setup_labelstudio(self, url, token):
        self.config_state["labelstudio"] = {"url": url, "token": token}
        config = eval(self.config_state.__repr__())
        with open(self.config_path, 'w') as fo:
            yaml.dump(config, fo)
        print(f"Setup Labelstudio url|{url} and token|{token} successfully!")
    
    def setup_cvat(self, url, username, password, org_name):
        self.config_state["cvat"] = {"url": url, "username": username, "password": password, "org_name": org_name}
        config = eval(self.config_state.__repr__())
        with open(self.config_path, 'w') as fo:
            yaml.dump(config, fo)
        print(f"Setup cvat url| {url}, username| {username}  password| {password} org_name| {org_name} successfully!")

