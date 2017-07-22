from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from prototxt import *

layer_dict = {'ConvNdBackward'   : 'Convolution',
              'ThresholdBackward': 'ReLU',
              'MaxPool2d'        : 'Pooling',
              'DropoutBackward'  : 'Dropout',
              'AddmmBackward'    : 'InnerProduct',
              'ViewBackward'     : 'Reshape'}

def pytorch2prototxt(model, x):
    net_info = OrderedDict()
    props = OrderedDict()
    props['name'] = 'pytorch'
    props['input'] = 'data'
    props['input_dim'] = x.size()
    
    layers = []

    var = model(x)

    def add_layer(func):
        parent_type = str(type(func).__name__)
        parent_name = parent_type+str(id(func))
        parent_bottoms = []
        parent_top = parent_name

        layer = OrderedDict()
        layer['name'] = parent_name
        layer['type'] = layer_dict[parent_type]
        if hasattr(func, 'next_functions'):
            for u in func.next_functions:
                if u[0] is not None:
                    child_type = str(type(u[0]).__name__)
                    child_name = child_type + str(id(u[0]))
                    if child_type != 'AccumulateGrad' and (parent_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                        top_name = add_layer(u[0])
                        parent_bottoms.append(top_name)

        if len(parent_bottoms) > 0:
            layer['bottom'] = parent_bottoms 
        else:
            layer['bottom'] = ['data']
        if parent_type == 'ConvNdBackward':
            weights = func.next_functions[1][0].variable
            conv_param = OrderedDict()
            conv_param['num_output'] = weights.size(0)
            conv_param['kernel_size'] = weights.size(2)
            conv_param['stride'] = func.stride[0]
            layer['convolution_param'] = conv_param
        #elif parent_type == 'ThresholdBackward':
        elif parent_type == 'MaxPool2d':
            pooling_param = OrderedDict()
            pooling_param['pool'] = 'MAX'
            pooling_param['kernel_size'] = func.kernel_size[0]
            pooling_param['stride'] = func.stride[0]
            layer['pooling_param']  = pooling_param
        elif parent_type == 'DropoutBackward':
            parent_top = parent_bottoms[0]
            dropout_param = OrderedDict()
            dropout_param['dropout_ratio'] = func.p
            layer['dropout_param'] = dropout_param
        elif parent_type == 'AddmmBackward':
            inner_product_param = OrderedDict()
            inner_product_param['num_output'] = func.next_functions[0][0].variable.size(0)
            layer['inner_product_param'] = inner_product_param
        elif parent_type == 'ViewBackward':
            parent_top = parent_bottoms[0]

        layer['top'] = parent_top
        if parent_type != 'ViewBackward':
            layers.append(layer)
        return parent_top
    
    add_layer(var.grad_fn)
    net_info['props'] = props
    net_info['layers'] = layers
    return net_info

if __name__ == '__main__':

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
    
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)
    
    #m = Net()
    #x = Variable(torch.rand(1,3,28,28))

    import torchvision

    m = torchvision.models.alexnet()
    print(m)
    x = Variable(torch.rand(1, 3, 227, 227))
    net_info = pytorch2prototxt(m, x) 
    print_prototxt(net_info)
