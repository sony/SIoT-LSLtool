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
import json
import os
import yaml
import glob
from lsl_tools.cli.utils.config_dict import CfgDict
from lsl_tools.tools import set_args
from copy import deepcopy


class LSLProcess:
    """ lsl tools project manager """
    def __init__(self, data_manager, data_worker):
        self.work_dir = os.getcwd()
        self.args = None
        self.command = None
        self.config_path = None
        self.config_state = None
        self.test_info = None
        self.default_cfg = ".config"
        self.config_path = os.path.join(self.work_dir, f"{self.default_cfg}.yaml")
        self.data_manager = data_manager(self.config_path)
        self.data_worker = data_worker()

    def __call__(self, args):
        """  Entrance of LSl tools

        First construct command instructions, second check project status, then execute lsl commands
        args:
            args (argparse.Namespace): command line parameter parser object
        """
        self.process_args(args)
        self.read_config_state()
        self.run()

    def process_args(self, args):
        """ construct command instructions

        process args form Namespace to dict format with key of command type in one of
        [create/ import/ auto-label/ ...] and the value is also a dict of optional command
        parameters [ labeled/ unlabeled/ name/ ...]
        exp:
            {"import":
                {"annotation": xxx, "image": path, "name": xxx}
            }

        args:
            args (argparse.Namespace): command line parameter parser object
        """
        self.command = args.__dict__.pop("command")
        if self.command == "auto-label":
            self.command = "auto_label"
        
        # labelstudio
        if self.command == "labelstudio-export":
            self.command = "labelstudio_export"
        if self.command == "labelstudio-setup":
            self.command = "labelstudio_setup"
        if self.command == "labelstudio-ml-start":
            self.command = "labelstudio_ml_start"

        # cvat
        if self.command == "cvat-setup":
            self.command = "cvat_setup"
        if self.command == "cvat-export":
            self.command = "cvat_export"
        if self.command == "cvat-inference":
            self.command = "cvat_inference"
        if self.command == "cvat-serverless":
            self.command = "cvat_serverless"

        if self.command == "slidewindows-setup":
            self.command = "slidewindows_setup"
        if self.command == "split-data":
            self.command = "split_data"
        self.args = CfgDict({self.command: args.__dict__})
        self.args_check()
        os.makedirs(args.voc_tmp, exist_ok=True)
        args.package_top = os.path.abspath(os.path.dirname(__file__) + '/../') # top folder of installed lsl package
        set_args(args.__dict__)

    def read_config_state(self):
        """ update current project status form dick (.config.yaml) to lsl process

        This operation will read info from project config file and save those info
        to an object data_manager which is responsible for maintaining data info in
        current lsl process.
        """
        self.proj_check()
        self.data_manager.update()

    def run(self):
        """ LSL command line running logics

         main interface of LSL command tool and models
        """
        self.args.task = dict(self.data_manager.config_state).get('task')
        if self.args.task not in ['fsod', "wsis", "fsis", "fsis_segonly"]:
            print("Please specify LSL task when calling `LSL create`")
            exit(0)

        if self.command == "import":
            self.update_source_state()

        if self.command == "auto_label":
            self.auto_label()

        if self.command == "preview":
            if self.args.task in ["fsod", "fsis", "fsis_segonly", "wsis"]:
                self.data_preview()
            else:
                print(F"Preview command doesn't support `{self.args.task}` task")

        if self.command == "export":
            self.data_export()

        if self.command == "labelstudio_setup":
            self.labelstudio_setup()

        if self.command == "labelstudio_export":
            self.labelstudio_export()

        if self.command == "labelstudio_ml_start":
            self.labelstudio_ml_start()

        if self.command == "ls":
            self.show_data_source()

        if self.command == "rm":
            self.remove_data_source()

        if self.command == "rename":
            self.rename_data_source()

        if self.command == "coco2weak":
            self.coco_to_weak()
        
        if self.command == "split_data":
            self.split_data()
        
        if self.command == "cvat_setup":
            self.cvat_setup()

        if self.command == "cvat_export":
            self.cvat_export()

        if self.command == "cvat_inference":
            self.cvat_inference()

        if self.command == "cvat_serverless":
            self.cvat_serverless()

        return

    def init_proj(self):
        """ initiate directory as a project by create a hidden `.config.yaml` file

        The initial info is organized as follow, which is a dict and supports extension
        by adding field in dict manner.
        """
        with open(self.config_path, 'w') as fo:
            init_info = {
                "init": self.config_path,
                "data_source": {
                    "labeled": {},
                    "unlabeled": {}
                },
                "registered": [],
                "task": self.args[self.command].task,
            }
            yaml.dump(init_info, fo)
        print("Program initialized ")
        exit(0)

    def update_source_state(self):
        """ update source info when execute import command

        """
        # assert not (self.args["import"].get("labeled") and self.args["import"].get("unlabeled")), \
        #     "please specify labeled and unlabeled data separately"

        if self.args["import"].get("labelstudio_project_id"):
            from lsl_tools.labelstudio.import_project import Import_LabelStudio
            labelstudio_url = self.data_manager.config_state.labelstudio.url
            api_key = self.data_manager.config_state.labelstudio.token
            output_root = f"{self.work_dir}/labelstudio_tmp"
            if not os.path.exists(output_root):
                os.mkdir(output_root)
            images, num_imgs, annotation_path = Import_LabelStudio(labelstudio_url, api_key).export_project(self.args["import"].get("labelstudio_project_id"), output_root)
            data_info = {
                "ann": annotation_path,
                "img": images,
                "fmt": self.args["import"].format,
                "num": num_imgs,
            }
            self.data_manager.add_data_src(data_info, labeled=True, name=self.args["import"].name[0])
        
        elif self.args["import"].get("cvat_task_id"):
            from lsl_tools.cvat.import_task import Import_CVAT
            cvat_url = self.data_manager.config_state.cvat.url
            username = self.data_manager.config_state.cvat.username
            password = self.data_manager.config_state.cvat.password
            output_root = f"{self.work_dir}/cvat_tmp/{self.args['import'].name[0]}"
            assert not os.path.exists(output_root), "This name: {self.args['import'].name[0]} of dataset has already been imported, please change the name."
            os.makedirs(output_root)
            
            images, num_imgs, annotation_path = Import_CVAT(cvat_url, username, password).export_task(self.args["import"].get("cvat_task_id"), output_root, task=self.args.task)
            data_info = {
                "ann": annotation_path,
                "img": images,
                "fmt": self.args["import"].format,
                "num": num_imgs,
            }
            self.data_manager.add_data_src(data_info, labeled=True, name=self.args["import"].name[0])


        elif self.args["import"].get("annotation"):
            with open(self.args["import"].annotation, 'r') as f:
                num_imgs = len(json.load(f)["images"])
            data_info = {
                "ann": self.args["import"].annotation,
                "img": self.args["import"].image,
                "fmt": self.args["import"].format,
                "num": num_imgs,
            }
            self.data_manager.add_data_src(data_info, labeled=True, name=self.args["import"].name[0])
        elif self.args["import"].get("weak_set"):
            weak_data = self.args["import"].get("weak_set")
            num_files = len(glob.glob(f"{weak_data}/*/*"))
            data_info = {
                "ann": self.args["import"].annotation,
                "img": self.args["import"].weak_set,
                "fmt": self.args["import"].format,
                "num": num_files,
            }
            self.data_manager.add_data_src(data_info, labeled=True, name=self.args["import"].name[0])
        else:
            data_info = {
                "img": self.args["import"].unlabeled_image,
                "fmt": self.args["import"].format,
                "num": len(os.listdir(self.args["import"].unlabeled_image))
            }
            self.data_manager.add_data_src(data_info, labeled=False, name=self.args["import"].name[0])

        self.data_manager.record_data_src()
        print("Data source updated ")

    def data_preview(self):
        # TODO: After first generated, using generated pseudo label file to draw preview images
        # self.data_worker.preview(
        #     self.data_manager.config_state.data_source,
        #     self.args.preview.name[0]
        # )

        self.data_worker.preview_coco_label(
            self.data_manager.config_state.data_source,
            self.args.preview.train,
            self.args.preview.name,
            self.args.task,
            self.args.preview.conf,
        )

    def auto_label(self):
        """ auto label task will assign to data_worker object """

        base_label_cmds = self.args.auto_label
        # slidewindow start
        if base_label_cmds.slidewindow:
            label_cmds = deepcopy(base_label_cmds)
            if self.args.task == "fsis_segonly":
                # cancel object thereshold filter
                label_cmds.object_threshold = 0.001
            data_infos, label_cmds = self.data_worker.slidewindow_data(
                label_cmds=label_cmds,
                data_info=self.data_manager.config_state.data_source,
                )
            self.data_manager.config_state.data_source = data_infos
        else:
            label_cmds = base_label_cmds

        if self.args.task == "wsis":
            self.data_worker.work_wsis(
            data_info=self.data_manager.config_state.data_source,
            test_name=label_cmds.test,
            train_names=label_cmds.train,
            valid_name=label_cmds.valid,
            iterations=label_cmds.iter,
            conf_thresh=label_cmds.conf
        )
        elif self.args.task == "fsod":
            self.data_worker.work_glip(
            data_info=self.data_manager.config_state.data_source,
            test_name=label_cmds.test,
            train_names=label_cmds.train,
            valid_name=label_cmds.valid,
            iterations=label_cmds.iter,
            conf_thresh=label_cmds.conf
        )
        elif self.args.task in ["fsis", "fsis_segonly"]:
            self.data_worker.work_sam(
            data_info=self.data_manager.config_state.data_source,
            test_name=label_cmds.test,
            train_names=label_cmds.train,
            valid_name=label_cmds.valid,
            task = self.args.task,
            iterations=label_cmds.iter,
            conf_thresh=label_cmds.conf
        )
        # restore slide windows
        if base_label_cmds.slidewindow and label_cmds.test:
            data_infos, label_cmds = self.data_worker.slidewindow_restore(
                label_cmds=base_label_cmds,
                cropped_label_cmds=label_cmds,
                data_info=self.data_manager.config_state.data_source,
                task_name=self.args.task,
                )


        if self.args.auto_label.test:
            self.data_manager.add_training_info(
                {"auto_label": self.data_worker.training_info},
                self.args.auto_label.test,
            )

    def show_data_source(self):
        self.data_manager.show_data_info()

    def remove_data_source(self):
        name = self.args.rm.name[0]
        self.data_manager.remove_data_src(name)

    def rename_data_source(self):
        old_name, new_name = self.args.rename.name
        self.data_manager.rename_data_src(old_name, new_name)

    def data_export(self):
        file_names = self.args.export.name
        # data_paths = self.data_manager.get_data_path(file_names)
        self.data_worker.export_ann_data(file_names, self.work_dir)
    
    def labelstudio_setup(self):
        url = self.args.labelstudio_setup.url
        token = self.args.labelstudio_setup.token
        # data_paths = self.data_manager.get_data_path(file_names)
        self.data_manager.setup_labelstudio(url, token)

    def labelstudio_export(self):
        file_names = self.args.labelstudio_export.name
        source_name = self.args.labelstudio_export.train[0]
        labelstudio_project_name = self.args.labelstudio_export.labelstudio_name
        # data_paths = self.data_manager.get_data_path(file_names)
        self.data_worker.export_to_labelstudio(
            config_state = self.data_manager.config_state, 
            source_name = source_name, 
            test_names = file_names, 
            project_name = labelstudio_project_name, 
            output_dir = self.work_dir, 
            task_name = self.args.task)
    
    def labelstudio_ml_start(self):
        source_name = self.args.labelstudio_ml_start.train[0]
        labelstudio_ml_port = self.args.labelstudio_ml_start.labelstudio_ml_port
        conf = self.args.labelstudio_ml_start.conf
        self.data_worker.labelstudio_ml_run(
            args= self.args,
            config_state = self.data_manager.config_state, 
            source_name = source_name,
            labelstudio_ml_port = labelstudio_ml_port,
            confidence_score = conf
            )

    def split_data(self):
        annotation_path = self.args.split_data.annotation
        split_ratio = self.args.split_data.split_ratio
        output_path = self.args.split_data.output_path
        self.data_worker.split_coco_data(
            input_path = annotation_path,
            output_path = output_path,
            split_ratio = split_ratio
        )
    
    def cvat_setup(self):
        url = self.args.cvat_setup.url
        username = self.args.cvat_setup.cvat_username
        password = self.args.cvat_setup.cvat_password
        org_name = self.args.cvat_setup.cvat_orgname
        # token = self.args.cvat_setup.token
        # data_paths = self.data_manager.get_data_path(file_names)
        self.data_manager.setup_cvat(url, username, password, org_name)

    def cvat_export(self):
        file_names = self.args.cvat_export.name
        source_name = self.args.cvat_export.train[0]
        cvat_task_name = self.args.cvat_export.cvat_name
        # data_paths = self.data_manager.get_data_path(file_names)
        self.data_worker.export_to_cvat(
            config_state = self.data_manager.config_state, 
            source_name = source_name, 
            test_names = file_names, 
            cvat_task_name = cvat_task_name, 
            output_dir = self.work_dir, 
            task_name = self.args.task)
    
    def cvat_inference(self):
        source_name = self.args.cvat_inference.train[0]
        task_id = self.args.cvat_inference.cvat_task_id
        conf = self.args.cvat_inference.conf
        self.data_worker.infer_cvat(
            args= self.args,
            config_state = self.data_manager.config_state, 
            source_name = source_name,
            task_id = task_id,
            confidence_score = conf
            )
        
    def cvat_serverless(self):
        source_name = self.args.cvat_serverless.train[0]
        conf = self.args.cvat_serverless.conf
        cvat_serverless_image = self.args.cvat_serverless.cvat_serverless_image
        self.data_worker.build_cvat_serverless(
            args= self.args,
            source_name = source_name,
            cvat_serverless_image = cvat_serverless_image,
            confidence_score = conf
            )

    def proj_check(self):
        """ Check whether a current directory is an lsl project

        By default, current dir is not an lsl project, which need to be initialized by `lsl create`
        """
        if not os.path.exists(self.config_path) and self.command == "create":
            self.init_proj()

        if not os.path.exists(self.config_path) and self.command != "create":
            print(
                "current directory may not be an lsl program, "
                "you need to initiate the directory by `lsl create` first"
            )
            exit(0)

        if os.path.exists(self.config_path) and self.command == "create":
            ans = input(
                "You have created a lsl project in this directory, "
                "if recreated, the project will be initiated\n"
                "Do you want to initiate the project? [y/n]"
            )
            if ans == 'y':
                self.init_proj()
            else:
                print("Action canceled ")
                exit(0)

    def args_check(self):
        if self.command == "create" and self.args[self.command].task is None:
            print("LSL task name should be specified by --task {fsod, wsis, fsis, sam_only}")
            exit(0)
        
        if self.command == "auto_label":
            cmd_args = self.args.auto_label
            # if (not cmd_args.test) or (not cmd_args.train):
            if not cmd_args.train: # ignore target if not specified
                print(
                    "when doing auto-label, please specify source data \n"
                    "by `--train`"
                )
                exit(0)
            if not cmd_args.valid:
                print("when doing auto-label, if validation datasest not specified by `--valid`, then validation AP"
                      " is not available for measurement")

        if self.command == "import" or self.command == "export" or self.command == "preview":
            if self.args[self.command].name is None:
                print("Operation name should be specified by --name/-n")
                exit(0)

        if self.command == "preview":
            if len(self.args.preview.name) > 1:
                print("Please specify one data source for previewing")
                exit(0)

        if self.command == "rm":
            if len(self.args.rm.name) > 1:
                print("Only support remove one source at a time now")
                exit(0)
