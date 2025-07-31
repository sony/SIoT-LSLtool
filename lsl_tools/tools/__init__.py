from lsl_tools.cli.utils.config_dict import CfgDict

lsl_args = CfgDict()

def set_args(args):
    global lsl_args
    lsl_args.update(args)
