import os
import argparse
import torch
from PIL import Image

from model import build_model
from model.engine import do_transfer_style
from config import get_cfg_defaults
from model.engine.hr_transfer_style import do_hr_transfer_style
from model.meta_arch import GramMSELoss, StyleTransfer
from util.logger import setup_logger
from util.prepare_vgg import prepare_vgg_weights

import torch.nn as nn


def get_model(cfg):
    # build vgg_model
    vgg_model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    vgg_model.to(device)
    # load model weights
    prepare_vgg_weights(cfg)
    vgg_model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS))
    for param in vgg_model.parameters():
        param.requires_grad = False

    # define layers, loss functions
    loss_layers = cfg.LOSS.STYLE_LAYERS + cfg.LOSS.CONTENT_LAYERS
    loss_functions = [GramMSELoss()] * len(cfg.LOSS.STYLE_LAYERS) + \
                     [nn.MSELoss()] * len(cfg.LOSS.CONTENT_LAYERS)
    loss_functions = [loss_function.to(device) for loss_function in loss_functions]

    # loss weights settings
    loss_weights = cfg.LOSS.STYLE_WEIGHTS + cfg.LOSS.CONTENT_WEIGHTS

    model = StyleTransfer(vgg_model, loss_layers, loss_functions, loss_weights)
    return model, device


def transfer_style(cfg, high_resolution=False):
    # build model
    model, device = get_model(cfg)

    # load images
    content_image = Image.open(cfg.DATA.CONTENT_IMG_PATH)
    style_image = Image.open(cfg.DATA.STYLE_IMG_PATH)

    # start transferring the style
    out_image = do_transfer_style(
        cfg,
        model,
        content_image,
        style_image,
        device
    )

    if high_resolution:
        do_hr_transfer_style(
            cfg,
            model,
            content_image,
            style_image,
            out_image,
            device
        )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Image Style Transfer Using Convolutional Neural Networks.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="file",
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()

    # build the config
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()

    # setup the logger
    if not os.path.isdir(cfg.OUTPUT.DIR):
        os.mkdir(cfg.OUTPUT.DIR)
    logger = setup_logger("style-transfer", cfg.OUTPUT.DIR, 'log')
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))

    # Transfer style to content
    transfer_style(cfg, high_resolution=True)


if __name__ == "__main__":
    main()
