# pytorch-caffe-darknet-convert
Convert between pytorch, caffe and darknet models. Caffe darknet models can be load directly by pytorch.
- [x] darknet2pytorch : use darknet.py to load darknet model directly
- [x] caffe2pytorch   : use caffenet.py to load caffe model directly
- [x] darknet2caffe
- [x] caffe2darknet
- [x] pytorch2caffe
- [x] pytorch2darknet : pytorch2caffe then caffe2darknet

# Convert pytorch -> caffe -> darknet
```
1. python main.py -a resnet50-pytorch --pretrained -e /home/xiaohang/ImageNet/
=> using pre-trained model 'resnet50-pytorch'
Test: [0/196]   Time 14.016 (14.016)    Loss 0.4863 (0.4863)    Prec@1 85.938 (85.938)  Prec@5 97.656 (97.656)
Test: [10/196]  Time 0.179 (1.616)      Loss 0.9623 (0.6718)    Prec@1 76.562 (82.919)  Prec@5 93.359 (95.561)
Test: [20/196]  Time 0.165 (1.152)      Loss 0.7586 (0.6859)    Prec@1 86.328 (82.738)  Prec@5 92.578 (95.424)
Test: [30/196]  Time 0.253 (1.061)      Loss 0.7881 (0.6409)    Prec@1 80.469 (84.073)  Prec@5 95.312 (95.804)
Test: [40/196]  Time 0.648 (0.973)      Loss 0.6530 (0.6863)    Prec@1 82.812 (82.336)  Prec@5 96.484 (95.798)
Test: [50/196]  Time 0.153 (0.938)      Loss 0.4764 (0.6844)    Prec@1 89.062 (82.207)  Prec@5 97.266 (95.910)
Test: [60/196]  Time 0.149 (0.908)      Loss 0.9198 (0.6984)    Prec@1 76.172 (81.807)  Prec@5 95.312 (95.959)
Test: [70/196]  Time 0.645 (0.903)      Loss 0.7103 (0.6851)    Prec@1 78.516 (82.042)  Prec@5 96.094 (96.072)
Test: [80/196]  Time 0.663 (0.884)      Loss 1.4683 (0.7112)    Prec@1 62.109 (81.520)  Prec@5 88.672 (95.737)
Test: [90/196]  Time 1.429 (0.881)      Loss 1.8474 (0.7593)    Prec@1 57.031 (80.460)  Prec@5 86.719 (95.261)
Test: [100/196] Time 0.195 (0.859)      Loss 1.1329 (0.8115)    Prec@1 68.359 (79.297)  Prec@5 91.797 (94.694)
Test: [110/196] Time 1.109 (0.859)      Loss 0.8606 (0.8358)    Prec@1 77.734 (78.790)  Prec@5 93.750 (94.457)
Test: [120/196] Time 0.153 (0.851)      Loss 1.2403 (0.8538)    Prec@1 69.922 (78.483)  Prec@5 87.500 (94.150)
Test: [130/196] Time 2.340 (0.851)      Loss 0.7038 (0.8877)    Prec@1 80.469 (77.612)  Prec@5 96.484 (93.831)
Test: [140/196] Time 0.139 (0.839)      Loss 1.0392 (0.9057)    Prec@1 74.609 (77.263)  Prec@5 91.797 (93.628)
Test: [150/196] Time 2.273 (0.839)      Loss 1.0445 (0.9234)    Prec@1 75.781 (76.930)  Prec@5 90.234 (93.385)
Test: [160/196] Time 0.153 (0.830)      Loss 0.6993 (0.9374)    Prec@1 86.328 (76.672)  Prec@5 94.141 (93.180)
Test: [170/196] Time 2.016 (0.831)      Loss 0.6132 (0.9542)    Prec@1 82.422 (76.263)  Prec@5 97.656 (93.012)
Test: [180/196] Time 0.926 (0.823)      Loss 1.2884 (0.9700)    Prec@1 69.531 (75.930)  Prec@5 92.969 (92.872)
Test: [190/196] Time 1.609 (0.821)      Loss 1.1864 (0.9686)    Prec@1 67.188 (75.920)  Prec@5 94.922 (92.899)
 * Prec@1 76.022 Prec@5 92.934
2. python pytorch2caffe.py 
3. python main.py -a resnet50-pytorch2caffe --pretrained -e /home/xiaohang/ImageNet/
=> using pre-trained model 'resnet50-pytorch2caffe'
load weights resnet50-pytorch2caffe.caffemodel
Loading caffemodel:  resnet50-pytorch2caffe.caffemodel
Test: [0/196]   Time 14.528 (14.528)    Loss 0.4863 (0.4863)    Prec@1 85.938 (85.938)  Prec@5 97.656 (97.656)
Test: [10/196]  Time 0.356 (1.678)      Loss 0.9623 (0.6718)    Prec@1 76.562 (82.919)  Prec@5 93.359 (95.561)
Test: [20/196]  Time 0.183 (1.206)      Loss 0.7586 (0.6859)    Prec@1 86.328 (82.738)  Prec@5 92.578 (95.424)
Test: [30/196]  Time 0.428 (1.112)      Loss 0.7881 (0.6409)    Prec@1 80.469 (84.073)  Prec@5 95.312 (95.804)
Test: [40/196]  Time 0.820 (1.022)      Loss 0.6530 (0.6863)    Prec@1 82.812 (82.336)  Prec@5 96.484 (95.798)
Test: [50/196]  Time 0.290 (0.978)      Loss 0.4764 (0.6844)    Prec@1 89.062 (82.207)  Prec@5 97.266 (95.910)
Test: [60/196]  Time 0.477 (0.941)      Loss 0.9198 (0.6984)    Prec@1 76.172 (81.807)  Prec@5 95.312 (95.959)
Test: [70/196]  Time 0.246 (0.927)      Loss 0.7103 (0.6851)    Prec@1 78.516 (82.042)  Prec@5 96.094 (96.072)
Test: [80/196]  Time 0.877 (0.910)      Loss 1.4683 (0.7112)    Prec@1 62.109 (81.520)  Prec@5 88.672 (95.737)
Test: [90/196]  Time 0.752 (0.906)      Loss 1.8474 (0.7593)    Prec@1 57.031 (80.460)  Prec@5 86.719 (95.261)
Test: [100/196] Time 0.156 (0.883)      Loss 1.1329 (0.8115)    Prec@1 68.359 (79.297)  Prec@5 91.797 (94.694)
Test: [110/196] Time 0.324 (0.882)      Loss 0.8606 (0.8358)    Prec@1 77.734 (78.790)  Prec@5 93.750 (94.457)
Test: [120/196] Time 0.486 (0.878)      Loss 1.2403 (0.8538)    Prec@1 69.922 (78.483)  Prec@5 87.500 (94.150)
Test: [130/196] Time 1.067 (0.871)      Loss 0.7038 (0.8877)    Prec@1 80.469 (77.612)  Prec@5 96.484 (93.831)
Test: [140/196] Time 0.261 (0.863)      Loss 1.0392 (0.9057)    Prec@1 74.609 (77.263)  Prec@5 91.797 (93.628)
Test: [150/196] Time 0.354 (0.852)      Loss 1.0445 (0.9234)    Prec@1 75.781 (76.930)  Prec@5 90.234 (93.385)
Test: [160/196] Time 0.152 (0.851)      Loss 0.6993 (0.9374)    Prec@1 86.328 (76.672)  Prec@5 94.141 (93.180)
Test: [170/196] Time 0.688 (0.842)      Loss 0.6132 (0.9542)    Prec@1 82.422 (76.263)  Prec@5 97.656 (93.012)
Test: [180/196] Time 0.244 (0.839)      Loss 1.2884 (0.9700)    Prec@1 69.531 (75.930)  Prec@5 92.969 (92.872)
Test: [190/196] Time 0.383 (0.834)      Loss 1.1864 (0.9686)    Prec@1 67.188 (75.920)  Prec@5 94.922 (92.899)
 * Prec@1 76.022 Prec@5 92.934
4. python caffe2darknet.py resnet50-pytorch2caffe.prototxt resnet50-pytorch2caffe.caffemodel resnet50-caffe2darknet.cfg resnet50-caffe2darknet.weights
5. python main.py -a resnet50-caffe2darknet --pretrained -e /home/xiaohang/ImageNet/        
=> using pre-trained model 'resnet50-caffe2darknet'
load weights from resnet50-caffe2darknet.weights
Test: [0/196]   Time 15.418 (15.418)    Loss 0.4863 (0.4863)    Prec@1 85.938 (85.938)  Prec@5 97.656 (97.656)
Test: [10/196]  Time 0.393 (1.760)      Loss 0.9623 (0.6718)    Prec@1 76.562 (82.919)  Prec@5 93.359 (95.561)
Test: [20/196]  Time 0.264 (1.241)      Loss 0.7586 (0.6859)    Prec@1 86.328 (82.738)  Prec@5 92.578 (95.424)
Test: [30/196]  Time 0.160 (1.123)      Loss 0.7881 (0.6409)    Prec@1 80.469 (84.073)  Prec@5 95.312 (95.804)
Test: [40/196]  Time 0.789 (1.020)      Loss 0.6530 (0.6863)    Prec@1 82.812 (82.336)  Prec@5 96.484 (95.798)
Test: [50/196]  Time 0.354 (0.983)      Loss 0.4764 (0.6844)    Prec@1 89.062 (82.207)  Prec@5 97.266 (95.910)
Test: [60/196]  Time 0.458 (0.946)      Loss 0.9198 (0.6984)    Prec@1 76.172 (81.807)  Prec@5 95.312 (95.959)
Test: [70/196]  Time 0.848 (0.936)      Loss 0.7103 (0.6851)    Prec@1 78.516 (82.042)  Prec@5 96.094 (96.072)
Test: [80/196]  Time 0.993 (0.918)      Loss 1.4683 (0.7112)    Prec@1 62.109 (81.520)  Prec@5 88.672 (95.737)
Test: [90/196]  Time 1.750 (0.911)      Loss 1.8474 (0.7593)    Prec@1 57.031 (80.460)  Prec@5 86.719 (95.261)
Test: [100/196] Time 0.160 (0.889)      Loss 1.1329 (0.8115)    Prec@1 68.359 (79.297)  Prec@5 91.797 (94.694)
Test: [110/196] Time 1.261 (0.883)      Loss 0.8606 (0.8358)    Prec@1 77.734 (78.790)  Prec@5 93.750 (94.457)
Test: [120/196] Time 0.667 (0.874)      Loss 1.2403 (0.8538)    Prec@1 69.922 (78.483)  Prec@5 87.500 (94.150)
Test: [130/196] Time 1.216 (0.867)      Loss 0.7038 (0.8877)    Prec@1 80.469 (77.612)  Prec@5 96.484 (93.831)
Test: [140/196] Time 0.166 (0.857)      Loss 1.0392 (0.9057)    Prec@1 74.609 (77.263)  Prec@5 91.797 (93.628)
Test: [150/196] Time 1.123 (0.850)      Loss 1.0445 (0.9234)    Prec@1 75.781 (76.930)  Prec@5 90.234 (93.385)
Test: [160/196] Time 0.161 (0.845)      Loss 0.6993 (0.9374)    Prec@1 86.328 (76.672)  Prec@5 94.141 (93.180)
Test: [170/196] Time 0.345 (0.837)      Loss 0.6132 (0.9542)    Prec@1 82.422 (76.263)  Prec@5 97.656 (93.012)
Test: [180/196] Time 1.152 (0.839)      Loss 1.2884 (0.9700)    Prec@1 69.531 (75.930)  Prec@5 92.969 (92.872)
Test: [190/196] Time 0.165 (0.829)      Loss 1.1864 (0.9686)    Prec@1 67.188 (75.920)  Prec@5 94.922 (92.899)
 * Prec@1 76.022 Prec@5 92.934
```
imagenet data is processed [as described here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset)

# Convert pytorch -> darknet -> caffe
### convert resnet50 from pytorch to darknet and then to caffe
```
1. python pytorch2darknet.py 
2. python main.py -a resnet50-darknet --pretrained -e /home/xiaohang/ImageNet/
=> using pre-trained model 'resnet50-darknet'
load weights from resnet50.weights
Test: [0/196]   Time 15.029 (15.029)    Loss 6.0965 (6.0965)    Prec@1 85.938 (85.938)  Prec@5 97.656 (97.656)
Test: [10/196]  Time 0.380 (1.716)      Loss 6.2165 (6.1346)    Prec@1 76.562 (82.919)  Prec@5 93.359 (95.561)
Test: [20/196]  Time 0.167 (1.205)      Loss 6.0981 (6.1388)    Prec@1 86.328 (82.738)  Prec@5 92.578 (95.424)
Test: [30/196]  Time 0.163 (1.100)      Loss 6.1633 (6.1244)    Prec@1 80.469 (84.073)  Prec@5 95.312 (95.804)
Test: [40/196]  Time 0.862 (1.009)      Loss 6.1777 (6.1473)    Prec@1 82.812 (82.336)  Prec@5 96.484 (95.798)
Test: [50/196]  Time 0.713 (0.965)      Loss 6.0856 (6.1510)    Prec@1 89.062 (82.207)  Prec@5 97.266 (95.910)
Test: [60/196]  Time 0.867 (0.936)      Loss 6.1982 (6.1557)    Prec@1 76.172 (81.807)  Prec@5 95.312 (95.959)
Test: [70/196]  Time 0.451 (0.917)      Loss 6.1979 (6.1513)    Prec@1 78.516 (82.042)  Prec@5 96.094 (96.072)
Test: [80/196]  Time 1.749 (0.909)      Loss 6.3671 (6.1568)    Prec@1 62.109 (81.520)  Prec@5 88.672 (95.737)
Test: [90/196]  Time 0.904 (0.892)      Loss 6.4027 (6.1684)    Prec@1 57.031 (80.460)  Prec@5 86.719 (95.261)
Test: [100/196] Time 0.463 (0.874)      Loss 6.3013 (6.1812)    Prec@1 68.359 (79.297)  Prec@5 91.797 (94.694)
Test: [110/196] Time 0.892 (0.868)      Loss 6.1719 (6.1863)    Prec@1 77.734 (78.790)  Prec@5 93.750 (94.457)
Test: [120/196] Time 0.162 (0.860)      Loss 6.2912 (6.1894)    Prec@1 69.922 (78.483)  Prec@5 87.500 (94.150)
Test: [130/196] Time 1.983 (0.862)      Loss 6.1764 (6.1982)    Prec@1 80.469 (77.612)  Prec@5 96.484 (93.831)
Test: [140/196] Time 0.163 (0.850)      Loss 6.2354 (6.2017)    Prec@1 74.609 (77.263)  Prec@5 91.797 (93.628)
Test: [150/196] Time 1.820 (0.845)      Loss 6.1851 (6.2053)    Prec@1 75.781 (76.930)  Prec@5 90.234 (93.385)
Test: [160/196] Time 0.166 (0.835)      Loss 6.1462 (6.2080)    Prec@1 86.328 (76.672)  Prec@5 94.141 (93.180)
Test: [170/196] Time 2.107 (0.836)      Loss 6.1428 (6.2130)    Prec@1 82.422 (76.263)  Prec@5 97.656 (93.012)
Test: [180/196] Time 0.863 (0.828)      Loss 6.3378 (6.2168)    Prec@1 69.531 (75.930)  Prec@5 92.969 (92.872)
Test: [190/196] Time 1.622 (0.827)      Loss 6.3392 (6.2167)    Prec@1 67.188 (75.920)  Prec@5 94.922 (92.899)
 * Prec@1 76.022 Prec@5 92.934
3. python darknet2caffe.py cfg/resnet50.cfg resnet50.weights resnet50-darknet2caffe.prototxt resnet50-darknet2caffe.caffemodel
4. python main.py -a resnet50-darknet2caffe --pretrained -e /home/xiaohang/ImageNet/ 
=> using pre-trained model 'resnet50-darknet2caffe'
load weights resnet50-darknet2caffe.caffemodel
Loading caffemodel:  resnet50-darknet2caffe.caffemodel
Test: [0/196]   Time 14.646 (14.646)    Loss 0.4863 (0.4863)    Prec@1 85.938 (85.938)  Prec@5 97.656 (97.656)
Test: [10/196]  Time 0.395 (1.705)      Loss 0.9623 (0.6718)    Prec@1 76.562 (82.919)  Prec@5 93.359 (95.561)
Test: [20/196]  Time 0.343 (1.213)      Loss 0.7586 (0.6859)    Prec@1 86.328 (82.738)  Prec@5 92.578 (95.424)
Test: [30/196]  Time 0.156 (1.095)      Loss 0.7881 (0.6409)    Prec@1 80.469 (84.073)  Prec@5 95.312 (95.804)
Test: [40/196]  Time 0.159 (0.989)      Loss 0.6530 (0.6863)    Prec@1 82.812 (82.336)  Prec@5 96.484 (95.798)
Test: [50/196]  Time 0.155 (0.959)      Loss 0.4764 (0.6844)    Prec@1 89.062 (82.207)  Prec@5 97.266 (95.910)
Test: [60/196]  Time 0.156 (0.921)      Loss 0.9198 (0.6984)    Prec@1 76.172 (81.807)  Prec@5 95.312 (95.959)
Test: [70/196]  Time 0.263 (0.911)      Loss 0.7103 (0.6851)    Prec@1 78.516 (82.042)  Prec@5 96.094 (96.072)
Test: [80/196]  Time 0.390 (0.887)      Loss 1.4683 (0.7112)    Prec@1 62.109 (81.520)  Prec@5 88.672 (95.737)
Test: [90/196]  Time 0.727 (0.887)      Loss 1.8474 (0.7593)    Prec@1 57.031 (80.460)  Prec@5 86.719 (95.261)
Test: [100/196] Time 0.160 (0.860)      Loss 1.1329 (0.8115)    Prec@1 68.359 (79.297)  Prec@5 91.797 (94.694)
Test: [110/196] Time 0.155 (0.857)      Loss 0.8606 (0.8358)    Prec@1 77.734 (78.790)  Prec@5 93.750 (94.457)
Test: [120/196] Time 0.301 (0.850)      Loss 1.2403 (0.8538)    Prec@1 69.922 (78.483)  Prec@5 87.500 (94.150)
Test: [130/196] Time 1.884 (0.850)      Loss 0.7038 (0.8877)    Prec@1 80.469 (77.612)  Prec@5 96.484 (93.831)
Test: [140/196] Time 0.155 (0.836)      Loss 1.0392 (0.9057)    Prec@1 74.609 (77.263)  Prec@5 91.797 (93.628)
Test: [150/196] Time 2.057 (0.835)      Loss 1.0445 (0.9234)    Prec@1 75.781 (76.930)  Prec@5 90.234 (93.385)
Test: [160/196] Time 0.157 (0.825)      Loss 0.6993 (0.9374)    Prec@1 86.328 (76.672)  Prec@5 94.141 (93.180)
Test: [170/196] Time 1.769 (0.826)      Loss 0.6132 (0.9542)    Prec@1 82.422 (76.263)  Prec@5 97.656 (93.012)
Test: [180/196] Time 0.995 (0.818)      Loss 1.2884 (0.9700)    Prec@1 69.531 (75.930)  Prec@5 92.969 (92.872)
Test: [190/196] Time 1.447 (0.815)      Loss 1.1864 (0.9686)    Prec@1 67.188 (75.920)  Prec@5 94.922 (92.899)
 * Prec@1 76.022 Prec@5 92.934
```

---
# Convert darknet to caffe
### convert tiny-yolo from darknet to caffe
```
1. download tiny-yolo-voc.weights : https://pjreddie.com/media/files/tiny-yolo-voc.weights
https://github.com/pjreddie/darknet/blob/master/cfg/tiny-yolo-voc.cfg
2. python darknet2caffe.py tiny-yolo-voc.cfg tiny-yolo-voc.weights tiny-yolo-voc.prototxt tiny-yolo-voc.caffemodel
3. download voc data and process according to https://github.com/marvis/pytorch-yolo2
python valid.py cfg/voc.data tiny-yolo-voc.prototxt tiny-yolo-voc.caffemodel
4. python scripts/voc_eval.py results/comp4_det_test_
VOC07 metric? Yes
AP for aeroplane = 0.6094
AP for bicycle = 0.6781
AP for bird = 0.4573
AP for boat = 0.3786
AP for bottle = 0.2081
AP for bus = 0.6645
AP for car = 0.6587
AP for cat = 0.6720
AP for chair = 0.3245
AP for cow = 0.4902
AP for diningtable = 0.5549
AP for dog = 0.5905
AP for horse = 0.6871
AP for motorbike = 0.6695
AP for person = 0.5833
AP for pottedplant = 0.2535
AP for sheep = 0.5374
AP for sofa = 0.4878
AP for train = 0.7004
AP for tvmonitor = 0.5754
Mean AP = 0.5391
5. python detect.py tiny-yolo-voc.prototxt tiny-yolo-voc.caffemodel data/dog.jpg 
```

### convert tiny-yolo from darknet to caffe without bn
```
1. python darknet.py tiny-yolo-voc.cfg tiny-yolo-voc.weights tiny-yolo-voc-nobn.cfg tiny-yolo-voc-nobn.weights
2. python darknet2caffe.py tiny-yolo-voc-nobn.cfg tiny-yolo-voc-nobn.weights tiny-yolo-voc-nobn.prototxt tiny-yolo-voc-nobn.caffemodel
3. python valid.py cfg/voc.data tiny-yolo-voc-nobn.prototxt tiny-yolo-voc-nobn.caffemodel
4. python scripts/voc_eval.py results/comp4_det_test_
VOC07 metric? Yes
AP for aeroplane = 0.6094
AP for bicycle = 0.6781
AP for bird = 0.4573
AP for boat = 0.3786
AP for bottle = 0.2081
AP for bus = 0.6645
AP for car = 0.6587
AP for cat = 0.6720
AP for chair = 0.3245
AP for cow = 0.4902
AP for diningtable = 0.5549
AP for dog = 0.5905
AP for horse = 0.6871
AP for motorbike = 0.6695
AP for person = 0.5833
AP for pottedplant = 0.2535
AP for sheep = 0.5374
AP for sofa = 0.4878
AP for train = 0.7004
AP for tvmonitor = 0.5754
Mean AP = 0.5391
5. python detect.py tiny-yolo-voc-nobn.prototxt tiny-yolo-voc-nobn.caffemodel data/dog.jpg 
```
---
# Convert caffe to darknet
### convert kaiming's resnet50 from caffe to darknet
```
1. download resnet50 from https://github.com/KaimingHe/deep-residual-networks
ResNet-50-deploy.prototxt: https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt
ResNet-50-model.caffemodel and ResNet_mean.binaryproto : https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777
python main.py -a resnet50-kaiming --pretrained -e /home/xiaohang/ImageNet/
=> using pre-trained model 'resnet50-kaiming'
load weights from ResNet-50-model.caffemodel
Loading caffemodel:  ResNet-50-model.caffemodel
convlution conv1 has bias
Test: [0/196]   Time 15.196 (15.196)    Loss 0.5485 (0.5485)    Prec@1 87.500 (87.500)  Prec@5 96.875 (96.875)
Test: [10/196]  Time 0.603 (1.851)      Loss 1.0412 (0.7676)    Prec@1 73.828 (80.717)  Prec@5 92.578 (94.567)
Test: [20/196]  Time 0.359 (1.361)      Loss 0.7243 (0.7656)    Prec@1 85.547 (80.804)  Prec@5 92.969 (94.494)
Test: [30/196]  Time 0.883 (1.240)      Loss 0.8022 (0.7273)    Prec@1 82.031 (81.981)  Prec@5 95.703 (94.796)
Test: [40/196]  Time 1.166 (1.159)      Loss 0.8016 (0.7690)    Prec@1 79.297 (80.516)  Prec@5 94.922 (94.779)
Test: [50/196]  Time 1.490 (1.108)      Loss 0.4365 (0.7552)    Prec@1 89.062 (80.668)  Prec@5 98.047 (95.044)
Test: [60/196]  Time 0.295 (1.072)      Loss 1.0453 (0.7689)    Prec@1 72.656 (80.277)  Prec@5 93.750 (95.210)
Test: [70/196]  Time 0.728 (1.064)      Loss 0.7959 (0.7542)    Prec@1 77.344 (80.573)  Prec@5 94.922 (95.384)
Test: [80/196]  Time 1.314 (1.047)      Loss 1.5740 (0.7775)    Prec@1 62.109 (80.179)  Prec@5 85.938 (95.100)
Test: [90/196]  Time 1.702 (1.040)      Loss 2.2488 (0.8350)    Prec@1 51.953 (78.979)  Prec@5 82.422 (94.463)
Test: [100/196] Time 0.300 (1.016)      Loss 1.2809 (0.8886)    Prec@1 69.141 (77.827)  Prec@5 89.844 (93.862)
Test: [110/196] Time 1.121 (1.011)      Loss 0.9445 (0.9139)    Prec@1 75.000 (77.404)  Prec@5 92.188 (93.567)
Test: [120/196] Time 0.667 (1.007)      Loss 1.4142 (0.9327)    Prec@1 66.406 (77.079)  Prec@5 86.328 (93.337)
Test: [130/196] Time 0.915 (0.995)      Loss 0.6773 (0.9680)    Prec@1 81.641 (76.273)  Prec@5 95.312 (92.981)
Test: [140/196] Time 0.315 (0.989)      Loss 1.1367 (0.9884)    Prec@1 74.219 (75.923)  Prec@5 91.016 (92.758)
Test: [150/196] Time 0.840 (0.979)      Loss 1.2445 (1.0101)    Prec@1 76.953 (75.492)  Prec@5 88.672 (92.454)
Test: [160/196] Time 0.324 (0.978)      Loss 0.9249 (1.0276)    Prec@1 80.078 (75.182)  Prec@5 90.234 (92.185)
Test: [170/196] Time 0.534 (0.967)      Loss 0.6927 (1.0442)    Prec@1 80.859 (74.737)  Prec@5 96.875 (91.954)
Test: [180/196] Time 0.298 (0.965)      Loss 1.3764 (1.0596)    Prec@1 66.406 (74.350)  Prec@5 91.797 (91.786)
Test: [190/196] Time 0.414 (0.962)      Loss 1.1433 (1.0589)    Prec@1 71.094 (74.317)  Prec@5 94.531 (91.823)
 * Prec@1 74.448 Prec@5 91.884
Kaiming: ResNet-50 24.7% 7.8% (shorter side=256)
2. python caffe2darknet.py ResNet-50-deploy.prototxt ResNet-50-model.caffemodel ResNet-50-model.cfg ResNet-50-model.weights
3. python main.py -a resnet50-kaiming-dk --pretrained -e /home/xiaohang/ImageNet/        
=> using pre-trained model 'resnet50-kaiming-dk'
load weights from ResNet-50-model.weights
Test: [0/196]   Time 14.963 (14.963)    Loss 0.5485 (0.5485)    Prec@1 87.500 (87.500)  Prec@5 96.875 (96.875)
Test: [10/196]  Time 0.939 (1.876)      Loss 1.0412 (0.7676)    Prec@1 73.828 (80.717)  Prec@5 92.578 (94.567)
Test: [20/196]  Time 0.331 (1.392)      Loss 0.7243 (0.7656)    Prec@1 85.547 (80.804)  Prec@5 92.969 (94.494)
Test: [30/196]  Time 1.910 (1.267)      Loss 0.8022 (0.7273)    Prec@1 82.031 (81.981)  Prec@5 95.703 (94.796)
Test: [40/196]  Time 0.352 (1.154)      Loss 0.8016 (0.7690)    Prec@1 79.297 (80.516)  Prec@5 94.922 (94.779)
Test: [50/196]  Time 1.606 (1.111)      Loss 0.4365 (0.7552)    Prec@1 89.062 (80.668)  Prec@5 98.047 (95.044)
Test: [60/196]  Time 0.714 (1.077)      Loss 1.0453 (0.7689)    Prec@1 72.656 (80.277)  Prec@5 93.750 (95.210)
Test: [70/196]  Time 0.332 (1.055)      Loss 0.7959 (0.7542)    Prec@1 77.344 (80.573)  Prec@5 94.922 (95.384)
Test: [80/196]  Time 1.654 (1.054)      Loss 1.5740 (0.7775)    Prec@1 62.109 (80.179)  Prec@5 85.938 (95.100)
Test: [90/196]  Time 0.344 (1.030)      Loss 2.2488 (0.8350)    Prec@1 51.953 (78.979)  Prec@5 82.422 (94.463)
Test: [100/196] Time 1.332 (1.016)      Loss 1.2809 (0.8886)    Prec@1 69.141 (77.827)  Prec@5 89.844 (93.862)
Test: [110/196] Time 0.336 (1.005)      Loss 0.9445 (0.9139)    Prec@1 75.000 (77.404)  Prec@5 92.188 (93.567)
Test: [120/196] Time 1.411 (1.000)      Loss 1.4142 (0.9327)    Prec@1 66.406 (77.079)  Prec@5 86.328 (93.337)
Test: [130/196] Time 1.784 (0.997)      Loss 0.6773 (0.9680)    Prec@1 81.641 (76.273)  Prec@5 95.312 (92.981)
Test: [140/196] Time 0.374 (0.986)      Loss 1.1367 (0.9884)    Prec@1 74.219 (75.923)  Prec@5 91.016 (92.758)
Test: [150/196] Time 1.725 (0.983)      Loss 1.2445 (1.0101)    Prec@1 76.953 (75.492)  Prec@5 88.672 (92.454)
Test: [160/196] Time 0.345 (0.974)      Loss 0.9249 (1.0276)    Prec@1 80.078 (75.182)  Prec@5 90.234 (92.185)
Test: [170/196] Time 1.802 (0.972)      Loss 0.6927 (1.0442)    Prec@1 80.859 (74.737)  Prec@5 96.875 (91.954)
Test: [180/196] Time 1.748 (0.967)      Loss 1.3764 (1.0596)    Prec@1 66.406 (74.350)  Prec@5 91.797 (91.786)
Test: [190/196] Time 1.099 (0.960)      Loss 1.1433 (1.0589)    Prec@1 71.094 (74.317)  Prec@5 94.531 (91.823)
 * Prec@1 74.448 Prec@5 91.884
```

### convert resnet18 from caffe to darknet
```
1. Download resnet18 from https://github.com/HolmesShuan/ResNet-18-Caffemodel-on-ImageNet.git and save as resnet-18.caffemodel
python main.py -a resnet18-caffe --pretrained -e /home/xiaohang/ImageNet/       
=> using pre-trained model 'resnet18-caffe'

load weights from resnet-18.caffemodel
Loading caffemodel:  resnet-18.caffemodel
Test: [0/196]   Time 14.473 (14.473)    Loss 0.6839 (0.6839)    Prec@1 83.594 (83.594)  Prec@5 95.703 (95.703)
Test: [10/196]  Time 0.313 (1.738)      Loss 1.3643 (1.0104)    Prec@1 62.500 (74.183)  Prec@5 89.844 (91.868)
Test: [20/196]  Time 0.198 (1.274)      Loss 1.1714 (1.0130)    Prec@1 75.391 (74.126)  Prec@5 87.891 (91.853)
Test: [30/196]  Time 0.205 (1.199)      Loss 1.0284 (0.9888)    Prec@1 74.609 (74.849)  Prec@5 92.969 (92.036)
Test: [40/196]  Time 0.354 (1.101)      Loss 0.9944 (1.0499)    Prec@1 73.047 (72.933)  Prec@5 95.703 (92.073)
Test: [50/196]  Time 0.451 (1.065)      Loss 0.6899 (1.0433)    Prec@1 86.719 (73.032)  Prec@5 95.312 (92.310)
Test: [60/196]  Time 0.430 (1.033)      Loss 1.1791 (1.0446)    Prec@1 66.406 (72.688)  Prec@5 92.578 (92.508)
Test: [70/196]  Time 0.487 (1.030)      Loss 1.0826 (1.0279)    Prec@1 71.875 (73.234)  Prec@5 90.625 (92.567)
Test: [80/196]  Time 0.832 (1.015)      Loss 1.8822 (1.0599)    Prec@1 57.422 (72.632)  Prec@5 82.812 (92.110)
Test: [90/196]  Time 2.098 (1.017)      Loss 2.2423 (1.1228)    Prec@1 48.047 (71.321)  Prec@5 78.906 (91.312)
Test: [100/196] Time 0.199 (0.999)      Loss 1.6156 (1.1888)    Prec@1 60.938 (70.073)  Prec@5 84.375 (90.447)
Test: [110/196] Time 1.784 (1.003)      Loss 1.2417 (1.2139)    Prec@1 69.922 (69.640)  Prec@5 88.281 (90.072)
Test: [120/196] Time 0.264 (1.010)      Loss 1.9448 (1.2463)    Prec@1 58.984 (69.183)  Prec@5 79.297 (89.550)
Test: [130/196] Time 2.169 (1.016)      Loss 1.1295 (1.2835)    Prec@1 73.047 (68.350)  Prec@5 90.234 (89.057)
Test: [140/196] Time 0.302 (1.015)      Loss 1.5492 (1.3093)    Prec@1 63.672 (67.855)  Prec@5 84.766 (88.683)
Test: [150/196] Time 2.651 (1.020)      Loss 1.5608 (1.3354)    Prec@1 69.141 (67.459)  Prec@5 82.031 (88.271)
Test: [160/196] Time 0.260 (1.020)      Loss 1.4529 (1.3561)    Prec@1 69.922 (67.151)  Prec@5 84.766 (87.976)
Test: [170/196] Time 2.067 (1.023)      Loss 1.1040 (1.3789)    Prec@1 69.922 (66.642)  Prec@5 91.016 (87.653)
Test: [180/196] Time 1.118 (1.021)      Loss 1.5395 (1.3954)    Prec@1 60.547 (66.313)  Prec@5 87.891 (87.412)
Test: [190/196] Time 1.031 (1.024)      Loss 1.6755 (1.3916)    Prec@1 56.641 (66.371)  Prec@5 84.766 (87.484)
 * Prec@1 66.562 Prec@5 87.562
2. python caffe2darknet.py cfg/resnet-18.prototxt resnet-18.caffemodel resnet-18.cfg resnet-18.weights
3. python main.py -a resnet18-darknet --pretrained -e /home/xiaohang/ImageNet/
=> using pre-trained model 'resnet18-darknet'
load weights from resnet-18.weights
Test: [0/196]   Time 15.171 (15.171)    Loss 0.6839 (0.6839)    Prec@1 83.594 (83.594)  Prec@5 95.703 (95.703)
Test: [10/196]  Time 0.560 (1.835)      Loss 1.3643 (1.0104)    Prec@1 62.500 (74.183)  Prec@5 89.844 (91.868)
Test: [20/196]  Time 0.290 (1.345)      Loss 1.1714 (1.0130)    Prec@1 75.391 (74.126)  Prec@5 87.891 (91.853)
Test: [30/196]  Time 1.594 (1.237)      Loss 1.0284 (0.9888)    Prec@1 74.609 (74.849)  Prec@5 92.969 (92.036)
Test: [40/196]  Time 0.820 (1.151)      Loss 0.9944 (1.0499)    Prec@1 73.047 (72.933)  Prec@5 95.703 (92.073)
Test: [50/196]  Time 0.928 (1.114)      Loss 0.6899 (1.0433)    Prec@1 86.719 (73.032)  Prec@5 95.312 (92.310)
Test: [60/196]  Time 0.264 (1.081)      Loss 1.1791 (1.0446)    Prec@1 66.406 (72.688)  Prec@5 92.578 (92.508)
Test: [70/196]  Time 0.694 (1.080)      Loss 1.0826 (1.0279)    Prec@1 71.875 (73.234)  Prec@5 90.625 (92.567)
Test: [80/196]  Time 0.742 (1.059)      Loss 1.8822 (1.0599)    Prec@1 57.422 (72.632)  Prec@5 82.812 (92.110)
Test: [90/196]  Time 0.920 (1.057)      Loss 2.2423 (1.1228)    Prec@1 48.047 (71.321)  Prec@5 78.906 (91.312)
Test: [100/196] Time 0.251 (1.035)      Loss 1.6156 (1.1888)    Prec@1 60.938 (70.073)  Prec@5 84.375 (90.447)
Test: [110/196] Time 0.857 (1.034)      Loss 1.2417 (1.2139)    Prec@1 69.922 (69.640)  Prec@5 88.281 (90.072)
Test: [120/196] Time 0.857 (1.029)      Loss 1.9448 (1.2463)    Prec@1 58.984 (69.183)  Prec@5 79.297 (89.550)
Test: [130/196] Time 1.816 (1.024)      Loss 1.1295 (1.2835)    Prec@1 73.047 (68.350)  Prec@5 90.234 (89.057)
Test: [140/196] Time 0.475 (1.010)      Loss 1.5492 (1.3093)    Prec@1 63.672 (67.855)  Prec@5 84.766 (88.683)
Test: [150/196] Time 1.124 (1.003)      Loss 1.5608 (1.3354)    Prec@1 69.141 (67.459)  Prec@5 82.031 (88.271)
Test: [160/196] Time 0.812 (1.000)      Loss 1.4529 (1.3561)    Prec@1 69.922 (67.151)  Prec@5 84.766 (87.976)
Test: [170/196] Time 0.328 (0.989)      Loss 1.1040 (1.3789)    Prec@1 69.922 (66.642)  Prec@5 91.016 (87.653)
Test: [180/196] Time 1.271 (0.988)      Loss 1.5395 (1.3954)    Prec@1 60.547 (66.313)  Prec@5 87.891 (87.412)
Test: [190/196] Time 0.292 (0.980)      Loss 1.6755 (1.3916)    Prec@1 56.641 (66.371)  Prec@5 84.766 (87.484)
 * Prec@1 66.562 Prec@5 87.562
```

---



