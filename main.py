import os
import argparse
import torch
from PIL import Image
from torch.autograd import Variable

from model import build_model
from data import ImageTransform
from model.engine import do_transfer_style
from config import get_cfg_defaults
from util.logger import setup_logger
from util.prepare_vgg import prepare_vgg_weights

def transfer_style(cfg):
    # build model
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    # load model weights
    prepare_vgg_weights(cfg)
    model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS))
    for param in model.parameters():
        param.requires_grad = False

    # build image transformer
    image_transformer = ImageTransform(cfg)

    # load images, ordered as [content_image, style_image]
    image_paths = [cfg.DATA.CONTENT_IMG_PATH, cfg.DATA.STYLE_IMG_PATH]
    images = [Image.open(image_path) for image_path in image_paths]
    images_transformed = [image_transformer.preparation(image) for image in images]
    images_transformed = [Variable(image.unsqueeze(0).to(device)) for image in images_transformed]

    content_image, style_image = images_transformed

    # start transferring the style
    do_transfer_style(
        cfg,
        model,
        image_transformer,
        content_image,
        style_image,
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
    transfer_style(cfg)


if __name__ == "__main__":
    main()
