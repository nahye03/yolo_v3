import os, sys
import numpy as np
import torch
import torch.nn as nn

from utils.tools import *

def make_conv_layer(layer_idx : int, modules : nn.Module, layer_info : dict, in_channel = int):
    filters = int(layer_info['filters']) #output channels
    size = int(layer_info['size']) #kernel size
    stride = int(layer_info['stride']) #stride
    pad = (size - 1) // 2
    modules.add_module('layer_' + str(layer_idx) + '_conv',
                       nn.Conv2d(in_channel, filters, size, stride, pad))
    
    if layer_info['batch_normalize'] == '1':
        modules.add_module('layer_' + str(layer_idx) + '_bn',
                           nn.BatchNorm2d(filters))
    
    if layer_info['activation'] == 'leaky':
        modules.add_module('layer_' + str(layer_idx) + '_act',
                           nn.LeakyReLU())
    elif layer_info['activation'] == 'relu':
        modules.add_module('layer_' + str(layer_idx) + '_act',
                           nn.ReLU())

def make_shortcut_layer(layer_idx : int, modules : nn.Module):
    modules.add_module('layer_' + str(layer_idx) + '_shortcut',
                       nn.Identity())

def make_route_layer(layer_idx : int, modules : nn.Module):
    modules.add_module('layer_' + str(layer_idx) + '_route',
                       nn.Identity())

def make_upsample_layer(layer_idx : int, modules : nn.Module, layer_info : dict):
    stride = int(layer_info['stride'])
    modules.add_module('layer_' +  str(layer_idx) + '_upsample',
                       nn.Upsample(scale_factor=stride, mode='nearest'))

    

class Darknet53(nn.Module):
    def __init__(self, cfg, param):
        super().__init__()
        self.batch = int(param['batch'])
        self.in_channels = int(param['in_channels'])
        self.in_width = int(param['in_width'])
        self.in_height = int(param['in_height'])
        self.n_classes = int(param['classes'])
        self.module_cfg = parse_model_config(cfg)
        self.module_list = self.set_layer(self.module_cfg)
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0] ,self.Yololayer)]
    
    class Yololayer(nn.Module):
        def __init__(self, layer_info : dict, in_width : int, in_height : int):
            super().__init__()
            self.n_classes = int(layer_info['classes'])
            self.ignore_thresh = float(layer_info['ignore_thresh']) #for bbox threshold
            self.box_attr = self.n_classes + 5 #box[4] + objectness[1] + class_prob[n_classes]
            mask_idxes = [int(x) for x in layer_info['mask'].split(',')]
            anchor_all = [int(x) for x in layer_info['anchors'].split(',')]
            anchor_all = [(anchor_all[i], anchor_all[i+1]) for i in range(0, len(anchor_all), 2)]
            self.anchor = torch.tensor([anchor_all[x] for x in mask_idxes]) #anchor shape : [3 anchor, 2(w,h)]
            self.in_width = in_width
            self.in_height = in_height
            self.stride = None
            self.lw = None
            self.lh = None
        
        def forward(self, x):
            #x is input. [N C H W]
            self.lw, self.lh = x.shape[3], x.shape[2]
            self.anchor = self.anchor.to(x.device) 
            self.stride = torch.tensor([torch.div(self.in_width, self.lw, rounding_mode='floor'),
                                    torch.div(self.in_height, self.lh, rounding_mode='floor')]).to(x.device)
            
            #if kitti data. n_classes is 8. C = (5 + 8) * 3 = 39
            #[batch, box_attribute * anchor, lh, lw] ex)[1, 39, 19, 19]

            #4dim [batch, box_attribute * anchor, lh, lw] -> 5 dim [batch, anchor, box_attribute, lh, lw]
            # -> [batch, anchor, lh, lw, box_attribute]
            x = x.view(-1, self.anchor.shape[0], self.box_attr, self.lh, self.lw).permute(0, 1, 3, 4, 2).contiguous()


            if not self.training:
                anchor_grid = self.anchor.view(1, -1, 1, 1, 2).to(x.device) #shape [1, 3, 1, 1, 2]
                grids = self._make_grid(nx = self.lw, ny = self.lh).to(x.device)

                #get output
                x[...,0:2] = (torch.sigmoid(x[...,0:2])+grids) * self.stride #center xy
                x[...,2:4] = torch.exp(x[...,2:4] * anchor_grid) # width height
                x[...,4:] = torch.sigmoid(x[...,4:]) #obj, conf
                x = x.view(x.shape[0], -1, self.box_attr) #[batch, anchor*lh*lw, 13]

            return x

        def _make_grid(self, nx, ny):
            #[0,0] [1,0] [2,0]
            #[0,1] [1,1] [2,1]
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def set_layer(self, layer_info):
        module_list = nn.ModuleList()
        in_channels = [self.in_channels] #first channels of input
        for layer_idx, info in enumerate(layer_info):
            modules = nn.Sequential()
            if info['type'] == 'convolutional':
                make_conv_layer(layer_idx, modules, info, in_channels[-1])
                in_channels.append(int(info['filters'])) #append output channel to use next input channel
            elif info['type'] == 'shortcut': #add (element add)
                make_shortcut_layer(layer_idx, modules)
                in_channels.append(in_channels[-1])
            elif info['type'] == 'route': #concat
                make_route_layer(layer_idx, modules)
                layers = [int(y) for y in info['layers'].split(',')]
                if len(layers) == 1:
                    in_channels.append(in_channels[layers[0]])
                elif len(layers) == 2:
                    in_channels.append(in_channels[layers[0]] + in_channels[layers[1]])
            elif info['type'] == 'upsample':
                make_upsample_layer(layer_idx, modules, info)
                in_channels.append(in_channels[-1])
            elif info['type'] == 'yolo':
                yololayer = self.Yololayer(info, self.in_width, self.in_height)
                modules.add_module('layer_' + str(layer_idx) + 'yolo', yololayer)
                in_channels.append(in_channels[-1])
            
            module_list.append(modules)
        return module_list

    def initialize_weights(self):
        #track all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1) #scale
                nn.init.constant_(m.bias, 0) #shift
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        yolo_result = []
        layer_result = []
        for idx, (name, layer) in enumerate(zip(self.module_cfg, self.module_list)):
            if name['type'] == 'convolutional':
                x = layer(x)
                layer_result.append(x)
            elif name['type'] == 'shortcut':
                x = x + layer_result[int(name['from'])]
                layer_result.append(x)
            elif name['type'] == 'yolo':
                yolo_x = layer(x)
                layer_result.append(yolo_x)
                yolo_result.append(yolo_x)
            elif name['type'] == 'upsample':
                x = layer(x)
                layer_result.append(x)
            elif name['type'] == 'route':
                layers = [int(x) for x in name['layers'].split(',')]
                x = torch.cat([layer_result[l] for l in layers], dim=1)
                layer_result.append(x)
        
        return yolo_result if self.training else torch.cat(yolo_result, dim=1) #[3, pred_boxes,anchor*lh*lw , 13(bbox attribute 5 + n_class 8)]


