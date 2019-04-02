import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self, cfg, pool='max'):
        super(VGG, self).__init__()
        self.cfg = cfg
        self.pool = pool
        self.layers = {}
        conv_layers_dict = self.cfg.MODEL.VGG.CONV_LAYERS_DICT[0]
        for layer_name in conv_layers_dict:
            self.layers[layer_name] = nn.Conv2d(in_channels=conv_layers_dict[layer_name]['in_channels'],
                                                out_channels=conv_layers_dict[layer_name]['out_channels'],
                                                kernel_size=conv_layers_dict[layer_name]['kernel'],
                                                padding=conv_layers_dict[layer_name]['padding'], )

        pool_layers_dict = self.cfg.MODEL.VGG.POOL_LAYERS_DICT[0]
        for layer_name in pool_layers_dict:
            if self.pool == 'max':
                self.layers[layer_name] = nn.MaxPool2d(kernel_size=pool_layers_dict[layer_name]['kernel_size'],
                                                       stride=pool_layers_dict[layer_name]['stride'], )

        self.conv1_1 = self.layers['conv1_1']
        self.conv1_2 = self.layers['conv1_2']
        self.conv2_1 = self.layers['conv2_1']
        self.conv2_2 = self.layers['conv2_2']
        self.conv3_1 = self.layers['conv3_1']
        self.conv3_2 = self.layers['conv3_2']
        self.conv3_3 = self.layers['conv3_3']
        self.conv3_4 = self.layers['conv3_4']
        self.conv4_1 = self.layers['conv4_1']
        self.conv4_2 = self.layers['conv4_2']
        self.conv4_3 = self.layers['conv4_3']
        self.conv4_4 = self.layers['conv4_4']
        self.conv5_1 = self.layers['conv5_1']
        self.conv5_2 = self.layers['conv5_2']
        self.conv5_3 = self.layers['conv5_3']
        self.conv5_4 = self.layers['conv5_4']

        self.forward_seq = self.cfg.MODEL.VGG.FORWARD_SEQ
        self.out_seq = self.cfg.MODEL.VGG.OUT_SEQ

    def forward(self, input, out_keys):
        if len(self.forward_seq) != len(self.out_seq):
            raise Exception("Forward and Output of layers of VGG must have the same length.")

        outputs = {}
        prev_out = input
        for i in range(len(self.forward_seq)):
            if self.forward_seq[i].find("conv") != -1:
                outputs[self.out_seq[i]] = F.relu(self.layers[self.forward_seq[i]](prev_out))
            elif self.forward_seq[i].find("pool") != -1:
                outputs[self.out_seq[i]] = self.layers[self.forward_seq[i]](prev_out)

            prev_out = outputs[self.out_seq[i]]

        return [outputs[key] for key in out_keys]
