# caffe-darknet-convert
python caffe2darknet.py ResNet-50-deploy.prototxt

# Todo
- [x] convert prototxt to cfg
- [x] print cfg nicely
- [ ] load caffemodel
- [ ] convert caffemodel to darknet weights

# convert resnet18
```
1. Download resnet18 from https://github.com/HolmesShuan/ResNet-18-Caffemodel-on-ImageNet.git and save as resnet-18.caffemodel
2. python caffe2darknet.py resnet-18.prototxt resnet-18.caffemodel resnet-18.cfg resnet-18.weights
3. python main.py -a resnet18 --pretrained -e /home/xiaohang/ImageNet/
=> using pre-trained model 'resnet18'
load weights from resnet-18.weights
Test: [0/196]   Time 14.573 (14.573)    Loss 2.5307 (2.5307)    Prec@1 46.484 (46.484)  Prec@5 69.922 (69.922)
Test: [10/196]  Time 0.384 (1.729)      Loss 1.9900 (2.1642)    Prec@1 48.438 (50.036)  Prec@5 81.641 (78.232)
Test: [20/196]  Time 0.292 (1.283)      Loss 1.6162 (1.8913)    Prec@1 68.359 (55.394)  Prec@5 83.203 (81.548)
Test: [30/196]  Time 0.420 (1.186)      Loss 1.4610 (1.8192)    Prec@1 63.672 (56.918)  Prec@5 91.016 (82.271)
Test: [40/196]  Time 0.926 (1.111)      Loss 2.1566 (1.9333)    Prec@1 44.531 (54.059)  Prec@5 80.469 (80.564)
Test: [50/196]  Time 0.154 (1.078)      Loss 1.1988 (1.9701)    Prec@1 68.359 (52.497)  Prec@5 93.359 (80.170)
Test: [60/196]  Time 0.280 (1.043)      Loss 1.7173 (1.9385)    Prec@1 57.422 (52.798)  Prec@5 83.594 (80.590)
Test: [70/196]  Time 0.516 (1.044)      Loss 1.3971 (1.8828)    Prec@1 64.453 (54.110)  Prec@5 87.500 (81.289)
Test: [80/196]  Time 0.765 (1.027)      Loss 2.1168 (1.8747)    Prec@1 55.078 (54.239)  Prec@5 78.125 (81.318)
Test: [90/196]  Time 1.136 (1.030)      Loss 2.4778 (1.8945)    Prec@1 41.016 (54.095)  Prec@5 72.656 (80.894)
Test: [100/196] Time 0.126 (1.006)      Loss 2.0996 (1.9239)    Prec@1 46.875 (53.697)  Prec@5 75.391 (80.411)
Test: [110/196] Time 1.061 (1.009)      Loss 1.5661 (1.9209)    Prec@1 63.672 (54.001)  Prec@5 83.984 (80.335)
Test: [120/196] Time 0.565 (1.001)      Loss 2.1030 (1.9204)    Prec@1 55.469 (54.358)  Prec@5 76.953 (80.220)
Test: [130/196] Time 2.475 (1.001)      Loss 1.2870 (1.9304)    Prec@1 70.703 (54.201)  Prec@5 87.891 (80.078)
Test: [140/196] Time 0.124 (0.987)      Loss 1.7190 (1.9237)    Prec@1 61.328 (54.446)  Prec@5 83.203 (80.156)
Test: [150/196] Time 2.323 (0.985)      Loss 1.7020 (1.9305)    Prec@1 64.062 (54.553)  Prec@5 80.078 (80.042)
Test: [160/196] Time 0.202 (0.975)      Loss 1.4141 (1.9326)    Prec@1 69.141 (54.612)  Prec@5 85.938 (80.049)
Test: [170/196] Time 2.498 (0.978)      Loss 1.1998 (1.9457)    Prec@1 69.922 (54.448)  Prec@5 92.188 (79.829)
Test: [180/196] Time 0.695 (0.967)      Loss 3.5546 (1.9594)    Prec@1 26.953 (54.299)  Prec@5 51.562 (79.588)
Test: [190/196] Time 2.084 (0.968)      Loss 2.2773 (2.0395)    Prec@1 46.094 (53.280)  Prec@5 78.516 (78.332)
 * Prec@1 53.136 Prec@5 78.124
```


