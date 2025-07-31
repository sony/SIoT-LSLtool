from .config_node import CfgNode as CN


_C = CN()
# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.

""" 
Following common steps to construct a trainer for experiments:
    step 1 : prepare dataset
    step 2 : build a dataloader
    step 3 : build a model
    step 4 : set a loss function
    step 5 : set a solver 
"""

# =============================================================================
#                       STEP 1 : DATASET CONFIG NODE
# =============================================================================
_C.DATASETS = CN()
# Specifier of dataset
_C.DATASETS.DATA = 'coco'
_C.DATASETS.DATA_ROOT = '../data'  
# The root of dataset
_C.DATASETS.TRAIN = 'coco_2017_train'
_C.DATASETS.VAL = 'coco_2017_val'
_C.DATASETS.TRAIN_ANN = ''
_C.DATASETS.VAL_ANN = ''
_C.DATASETS.SHUFFLE = True  
# whether shuffle the dataset
_C.DATASETS.SYNTHETIC = False 
# whether synthetic the  dataset
_C.DATASETS.IMAGE_SIZE = 1024
_C.DATASETS.NOISED_BBOX = True
# whether 
_C.DATASETS.NOISED_RATIO = (0.64, 1.44)
_C.DATASETS.PROMPT = ''

_C.DATASETS.CROP_SIZE = (1024, 1024)
_C.DATASETS.RESIZE_SIZE = None
_C.DATASETS.PADDING = False
_C.DATASETS.MOSAIC = False


# =============================================================================
#                      STEP 2 : DATALOADER CONFIG NODE
# =============================================================================

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.IMS_PER_BATCH = 1



# =============================================================================
#                      STEP 3 : MODEL CONFIG NODE
# =============================================================================
_C.MODEL = CN()

_C.MODEL.NET = 'sam' 
# in ['sam', 'sam-adapter',]
_C.MODEL.DIM = 512
_C.MODEL.DEPTH = 1
_C.MODEL.HEADS = 16
_C.MODEL.PATCH_SIZE = 2
_C.MODEL.MLP_DIM = 1024
_C.MODEL.CHUNK = 96
_C.MODEL.ROI_SIZE = 96
_C.MODEL.EVL_CHUNK = None
_C.MODEL.FREEZE = True
_C.MODEL.SAM_ONLY = False



# =============================================================================
#                      STEP 4 : LOSS CONFIG NODE
# =============================================================================
_C.CRITERION = CN()
_C.CRITERION.NAME = "default_criterion"


# =============================================================================
#                      STEP 5 : SOLVER CONFIG NODE
# =============================================================================
_C.SOLVER = CN()
_C.SOLVER.SAM_CKPT = './checkpoint/sam/sam_vit_b_01ec64.pth'
_C.SOLVER.WEIGHT = ''
_C.SOLVER.OUT_SIZE = 1024
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.VAL_STEP = 10
_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.NUM_SAMPLE = 4 
# sample pos and neg
_C.SOLVER.MODE = 'bbox'  
# in ['bbox', 'point', 'bbox_point']
_C.SOLVER.DENSITY = 'centriod'  
# in ['centriod', 'center']
_C.SOLVER.THRESHOLD = 0.9
# the threshold of pred mask
_C.SOLVER.IOU_THRESHOLD = 0.5  
# the threshold of nms
_C.SOLVER.STEP = [30, 40]
_C.SOLVER.GAMMA = 0.1


# =============================================================================
#                      STEP 8 : config callbacks
# =============================================================================
_C.LOG_DIR = 'runs'
_C.EXP_NAME = 'new_project'
# _C.GPU = True
# _C.GPU_DEVICE = 0
_C.VIS = True
# whether vis the  dataset
# _C.DISTRIBUTE = None 
# multi GPU ids to use

def get_default_cfg() -> CN:
    """
    Get a copy of the default config.

    Returns:
        a CfgNode instance.
    """
    # from .defaults import _C

    return _C.clone()