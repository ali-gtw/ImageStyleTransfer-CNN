import os
from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = CN()

# ---------------------------------------------------------------------------- #
# Model Configs
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = 'VGG'
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHTS = "./models/vgg_conv.pth"  # should be a path to pth or ckpt file

# ---------------------------------------------------------------------------- #
# __VGG Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.VGG = CN()
_C.MODEL.VGG.CONV_LAYERS_DICT = {
    'conv1_1': {'in_channels': 3, 'out_channels': 64, 'kernel': 3, 'padding': 1, },
    'conv1_2': {'in_channels': 64, 'out_channels': 64, 'kernel': 3, 'padding': 1, },
    'conv2_1': {'in_channels': 64, 'out_channels': 128, 'kernel': 3, 'padding': 1, },
    'conv2_2': {'in_channels': 128, 'out_channels': 128, 'kernel': 3, 'padding': 1, },
    'conv3_1': {'in_channels': 128, 'out_channels': 256, 'kernel': 3, 'padding': 1, },
    'conv3_2': {'in_channels': 256, 'out_channels': 256, 'kernel': 3, 'padding': 1, },
    'conv3_3': {'in_channels': 256, 'out_channels': 256, 'kernel': 3, 'padding': 1, },
    'conv3_4': {'in_channels': 256, 'out_channels': 256, 'kernel': 3, 'padding': 1, },
    'conv4_1': {'in_channels': 256, 'out_channels': 512, 'kernel': 3, 'padding': 1, },
    'conv4_2': {'in_channels': 512, 'out_channels': 512, 'kernel': 3, 'padding': 1, },
    'conv4_3': {'in_channels': 512, 'out_channels': 512, 'kernel': 3, 'padding': 1, },
    'conv4_4': {'in_channels': 512, 'out_channels': 512, 'kernel': 3, 'padding': 1, },
    'conv5_1': {'in_channels': 512, 'out_channels': 512, 'kernel': 3, 'padding': 1, },
    'conv5_2': {'in_channels': 512, 'out_channels': 512, 'kernel': 3, 'padding': 1, },
    'conv5_3': {'in_channels': 512, 'out_channels': 512, 'kernel': 3, 'padding': 1, },
    'conv5_4': {'in_channels': 512, 'out_channels': 512, 'kernel': 3, 'padding': 1, },

}
_C.MODEL.VGG.POOL_LAYERS_DICT = {
    'pool_1': {'kernel_size': 2, 'stride': 2, },
    'pool_2': {'kernel_size': 2, 'stride': 2, },
    'pool_3': {'kernel_size': 2, 'stride': 2, },
    'pool_4': {'kernel_size': 2, 'stride': 2, },
    'pool_5': {'kernel_size': 2, 'stride': 2, },
}

_C.MODEL.VGG.FORWARD_SEQ = [
    'conv1_1', 'conv1_2', 'pool_1',
    'conv2_1', 'conv2_2', 'pool_2',
    'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'pool_3',
    'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'pool_4',
    'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'pool_5',
]

_C.MODEL.VGG.OUT_SEQ = [
    'relu1_1', 'relu1_2', 'pool_1',
    'relu2_1', 'relu2_2', 'pool_2',
    'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4', 'pool_3',
    'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4', 'pool_4',
    'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4', 'pool_5',
]


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
