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
        conv_layers_dict = self.cfg.MODEL.VGG.CONV_LAYERS_DICT
        for layer_name in conv_layers_dict:
            self.layers[layer_name] = nn.Conv2d(in_channels=conv_layers_dict[layer_name]['in_channels'],
                                                out_channels=conv_layers_dict[layer_name]['out_channels'],
                                                kernel_size=conv_layers_dict[layer_name]['kernel'],
                                                padding=conv_layers_dict[layer_name]['padding'], )

        pool_layers_dict = self.cfg.MODEL.VGG.POOL_LAYERS_DICT
        for layer_name in pool_layers_dict:
            if self.pool == 'max':
                self.layers[layer_name] = nn.MaxPool2d(kernel_size=pool_layers_dict[layer_name]['kernel_size'],
                                                       stride=pool_layers_dict[layer_name]['stride'], )

        self.forward_seq = self.cfg.MODEL.VGG.FORWARD_SEQ
        self.out_seq = self.cfg.MODEL.VGG.OUT_SEQ

    def forward(self, input, out_keys):
        if len(self.forward_seq) != len(self.out_seq):
            raise Exception("Forward and Output of layers of VGG must have the same length.")

        out_puts = {}
        prev_out = input
        for i in range(len(self.forward_seq)):
            if self.forward_seq[i].find("conv") != -1:
                out_puts[self.out_seq[i]] = F.relu(self.layers[self.forward_seq[i]](prev_out))
            elif self.forward_seq[i].find("pool") != -1:
                out_puts[self.out_seq[i]] = self.layers[self.forward_seq[i]](prev_out)

            prev_out = out_puts[self.out_seq[i]]

        return [out_puts[key] for key in out_keys]
