import os
import subprocess


def prepare_vgg_weights(cfg):
    if not os.path.isdir(cfg.MODEL.MODELS_DIR):
        subprocess.call('bash ./download_models.sh', shell=True)
    elif not os.path.isfile(cfg.MODEL.WEIGHTS):
        subprocess.call('bash ./download_models.sh', shell=True)
