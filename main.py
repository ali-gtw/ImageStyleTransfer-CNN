import os
import argparse
from datetime import datetime
import torch

from config import get_cfg_defaults


def main():


    # build the config
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()

if __name__ == "__main__":
    main()