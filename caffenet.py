import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from prototxt import *

class FCView(nn.Module):
    def __init__(self):
        super(FCView, self).__init__()

    def forward(self, x):
        nB = x.data.size(0)
        x = x.view(nB,-1)
        return x
    def __repr__(self):
        return 'view(nB, -1)'

class Eltwise(nn.Module):
    def __init__(self, operation='+'):
        super(Eltwise, self).__init__()
        self.operation = operation
    def forward(self, x1, x2):
        if isinstance(input_feats, tuple):
            print "error : The input of Eltwise layer must be a tuple"
        for i, feat in enumerate(input_feats):
            if x is None:
                x = feat
                continue
            if self.operation == '+' or self.operation == 'SUM':
                x += feat
            if self.operation == '*' or self.operation == 'MUL':
                x *= feat
            if self.operation == '/' or self.operation == 'DIV':
                x /= feat
        return x

class CaffeNet(nn.Module):
    def __init__(self, protofile):
        super(CaffeNet, self).__init__()
        self.net_info = parse_prototxt(protofile)
        self.models, self.loss = self.create_network(self.net_info)
        self.modelList = nn.ModuleList()
        for name,model in self.models.items():
            self.modelList.append(model)

    def forward(self, data):
        blobs = OrderedDict()
        blobs['data'] = data
        
        layers = self.net_info['layers']
        layer_num = len(layers)
        i = 0
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            ltype = layer['type']
            tname = layer['top']
            bname = layer['bottom']
            if ltype == 'Data' or ltype == 'Accuracy' or ltype == 'SoftmaxWithLoss':
                i = i + 1
                continue
            elif ltype == 'BatchNorm':
                i = i + 1
                tname = layers[i]['top']

            if ltype != 'Eltwise':
                bdata = blobs[bname]
                tdata = self.models[lname](bdata)
                blobs[tname] = tdata
            else:
                bdata0 = blobs[bname[0]]
                bdata1 = blobs[bname[1]]
                tdata = self.models[lname](bdata0, bdata1)
                blobs[tname] = tdata
        return blobs.values()[len(blobs)-1]

    def print_network(self):
        print(self.modelList)
        print_prototxt(self.net_info)

    def load_weights(self, caffemodel):
        model = parse_caffemodel(caffemodel)
        layers = model.layer
        if len(layers) == 0:
            print('Using V1LayerParameter')
            layers = model.layers

        lmap = {}
        for l in layers:
            lmap[l.name] = l

        layers = net_info['layers']
        layer_num = len(layers)
        i = 0
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            ltype = layer['type']
            if ltype == 'Convolution':
                self.models[lname].weight.data = torch.from_numpy(np.array(lmap[lname].blobs[0].data))
                if len(lmap[lname].blobs) > 1:
                    self.models[lname].bias.data = torch.from_numpy(np.array(lmap[lname].blobs[1].data))
                i = i + 1
            elif ltype == 'BatchNorm':
                scale_layer = layers[i+1]
                self.models[lname].running_mean = torch.from_numpy(np.arrray(lmap[lname].blobs[0].data) / lmap[lname].blobs[2].data[0])
                self.models[lname].running_var = torch.from_numpy(np.arrray(lmap[lname].blobs[1].data) / lmap[lname].blobs[2].data[0])
                self.models[lname].weight.data = torch.from_numpy(np.array(lmap[scale_layer['name']].blobs[0].data))
                self.models[lname].bias.data = torch.from_numpy(np.array(lmap[scale_layer['name']].blobs[1].data))
                i = i + 2
            elif ltype == 'InnerProduct':
                self.models[lname].weight.data = torch.from_numpy(np.array(lmap[lname].blobs[0].data))
                if len(lmap[lname].blobs) > 1:
                    self.models[lname].bias.data = torch.from_numpy(np.array(lmap[lname].blobs[1].data))
                i = i + 1
            elif ltype == 'Pooling' or ltype == 'Eltwise' or ltype == 'ReLU':
                i = i + 1
            else:
                print('load_weights: unknown type %s' % ltype)
                i = i + 1

    def create_network(self, net_info):
        models = OrderedDict()
        loss = None
        blob_channels = dict()
        blob_width = dict()
        blob_height = dict()

        blob_channels['data'] = 1
        blob_width['data'] = 28
        blob_height['data'] = 28

        layers = net_info['layers']
        props = net_info['props']
        layer_num = len(layers)
        i = 0
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            ltype = layer['type']
            if ltype == 'Data':
                i = i + 1
                continue
            bname = layer['bottom']
            tname = layer['top']
            #print('lname = %s, ltype = #%s#, bname = %s, tname = %s' % (lname, ltype, bname, tname))
            if ltype == 'Convolution':
                convolution_param = layer['convolution_param']
                channels = blob_channels[bname]
                out_filters = int(convolution_param['num_output'])
                kernel_size = int(convolution_param['kernel_size'])
                stride = int(convolution_param['stride']) if convolution_param.has_key('stride') else 1
                pad = int(convolution_param['pad']) if convolution_param.has_key('pad') else 0
                group = int(convolution_param['group']) if convolution_param.has_key('group') else 1
                bias = True
                if convolution_param['bias_term'] == 'false':
                    bias = False
                models[lname] = nn.Conv2d(channels, out_filters, kernel_size, stride,pad,group, bias=bias)
                blob_channels[tname] = out_filters
                blob_width[tname] = (blob_width[bname] + 2*pad - kernel_size)/stride + 1
                blob_height[tname] = (blob_height[bname] + 2*pad - kernel_size)/stride + 1
                i = i + 1
            elif ltype == 'BatchNorm':
                assert(i + 1 < layer_num)
                assert(layers[i+1]['type'] == 'Scale')
                momentum = layer['batch_norm_param']['moving_average_fraction']
                channels = blob_channels[bname]
                models[lname] = nn.BatchNorm2d(channels, momentum=momentum)
                tname = layers[i+1]['top']
                blob_channels[tname] = channels
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 2
            elif ltype == 'ReLU':
                inplace = (bname == tname)
                models[lname] = nn.ReLU(inplace=inplace)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'Pooling':
                kernel_size = int(layer['pooling_param']['kernel_size'])
                stride = int(layer['pooling_param']['stride'])
                models[lname] = nn.MaxPool2d(kernel_size, stride)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]/stride
                blob_height[tname] = blob_height[bname]/stride
                i = i + 1
            elif ltype == 'Eltwise':
                operation = layer['eltwise_param']['operation']
                bname0 = bname[0]
                bname1 = bname[1]
                models[lname] = Eltwise(operation)
                blob_channels[tname] = blob_channels[bname0]
                blob_width[tname] = blob_width[bname0]
                blob_height[tname] = blob_height[bname0]
                i = i + 1
            elif ltype == 'InnerProduct':
                filters = int(layer['inner_product_param']['num_output'])
                if blob_width[bname] != 1 or blob_height[bname] != 1:
                    channels = blob_channels[bname] * blob_width[bname] * blob_height[bname]
                    models[lname] = nn.Sequential(FCView(), nn.Linear(channels, filters))
                else:
                    channels = blob_channels[bname]
                    models[lname] = nn.Linear(channels, filters)
                blob_channels[tname] = filters
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'Softmax':
                models[lname] = nn.Softmax()
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'SoftmaxWithLoss':
                loss = nn.CrossEntropyLoss()
                i = i + 1
            else:
                print('create_network: unknown type #%s#' % ltype)
                i = i + 1
        return models, loss

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Usage: python caffenet.py model.prototxt')
        exit()
    from torch.autograd import Variable
    protofile = sys.argv[1]
    net = CaffeNet(protofile)
    net.print_network()
