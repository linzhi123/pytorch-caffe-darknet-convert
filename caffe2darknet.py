from collections import OrderedDict
from cfg import *

def prototxt2cfg(protofile):
    blocks = []
    block = OrderedDict()
    block['type'] = 'net'
    block['batch'] = '128'
    block['subdivisions'] = '8'
    block['height'] = '224'
    block['width'] = '224'
    block['channels'] = '256'
    block['momentum'] = '0.9'
    block['decay'] = '0.0001'
    block['learning_rate'] = '0.05'
    block['policy'] = 'poly'
    block['power'] = '4'
    block['max_batches'] = 500000
    blocks.append(block)

    net_info = parse_prototxt(protofile)
    layers = net_info['layers']
    layer_num = len(layers)
    i = 0 # layer id
    layer_id = dict()
    layer_id['data'] = 0
    while i < layer_num:
        layer = layers[i]
        if layer['type'] == 'Convolution':
            if layer_id[layer['bottom']] != len(blocks)-1:
                block = OrderedDict()
                block['type'] = 'route'
                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
                blocks.append(block)
            assert(i+1 < layer_num and layers[i+1]['type'] == 'BatchNorm')
            assert(i+2 < layer_num and layers[i+2]['type'] == 'Scale')
            conv_layer = layers[i]
            bn_layer = layers[i+1]
            scale_layer = layers[i+2]
            block = OrderedDict()
            block['type'] = 'convolutional'
            block['batch_normalize'] = '1'
            block['filters'] = conv_layer['convolution_param']['num_output']
            block['size'] = conv_layer['convolution_param']['kernel_size']
            block['stride'] = conv_layer['convolution_param']['stride']
            block['pad'] = '1'
            if i+3 < layer_num and layers[i+3]['type'] == 'ReLU':
                act_layer = layers[i+3]
                block['activation'] = 'relu'
                top = act_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 4
            else:
                block['activation'] = 'linear'
                top = scale_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 3
        elif layer['type'] == 'Pooling':
            assert(layer_id[layer['bottom']] == len(blocks)-1)
            block = OrderedDict()
            if layer['pooling_param']['pool'] == 'AVE':
                block['type'] = 'avgpool'
            elif layer['pooling_param']['pool'] == 'MAX':
                block['type'] = 'maxpool'
                block['size'] = layer['pooling_param']['kernel_size']
                block['stride'] = layer['pooling_param']['stride']
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1
        elif layer['type'] == 'Eltwise':
            bottoms = layer['bottom']
            bottom1 = layer_id[bottoms[0]] - len(blocks)
            bottom2 = layer_id[bottoms[1]] - len(blocks)
            assert(bottom1 == -1 or bottom2 == -1)
            from_id = bottom2 if bottom1 == -1 else bottom1
            block = OrderedDict()
            block['type'] = 'shortcut'
            block['from'] = str(from_id)
            assert(i+1 < layer_num and layers[i+1]['type'] == 'ReLU')
            block['activation'] = 'relu'
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 2
        elif layer['type'] == 'InnerProduct':
            assert(layer_id[layer['bottom']] == len(blocks)-1)
            block = OrderedDict()
            block['type'] = 'connected'
            block['output'] = layer['inner_product_param']['num_output']
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1
        elif layer['type'] == 'Softmax':
            assert(layer_id[layer['bottom']] == len(blocks)-1)
            block = OrderedDict()
            block['type'] = 'softmax'
            block['groups'] = 1
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1
    return blocks

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('try:')
        print('  python caffe2darknet.py ResNet-50-deploy.prototxt')
        exit()
    blocks = prototxt2cfg(sys.argv[1])
    print_cfg(blocks)
    print_cfg_nicely(blocks)
