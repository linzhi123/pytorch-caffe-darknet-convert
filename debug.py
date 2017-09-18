from __future__ import print_function
import torch.optim as optim
import os
import torch
import numpy as np
from darknet import Darknet
from caffenet import CaffeNet
from PIL import Image
from utils import image2torch, convert2cpu
from torch.autograd import Variable

cfgfile1 = 'reid.cfg'
weightfile1 = 'reid.weights'
cfgfile2 = 'reid_nbn.cfg'
weightfile2 = 'reid_nbn.weights'
cfgfile3 = 'reid_nbn.prototxt'
weightfile3 = 'reid_nbn.caffemodel'

m1 = Darknet(cfgfile1)
m1.load_weights(weightfile1)
m1.eval()

m2 = Darknet(cfgfile2)
m2.load_weights(weightfile2)
m2.eval()

m3 = CaffeNet(cfgfile3)
m3.load_weights(weightfile3)
m3.eval()

img = torch.rand(8,3,128, 64)
img = Variable(img)

output1 = m1(img).clone()
output2 = m2(img).clone()
output3 = m3(img).clone()
print('----- output1 ------------------')
print(output1.data.storage()[0:100])
print('----- output2 ------------------')
print(output2.data.storage()[0:100])
print('----- output3 ------------------')
print(output3.data.storage()[0:100])

