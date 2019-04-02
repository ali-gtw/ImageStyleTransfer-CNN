#!/usr/bin/env bash
mkdir models && cd models
wget -c --no-check-certificate https://bethgelab.org/media/uploads/pytorch_models/vgg_conv.pth
cd ..