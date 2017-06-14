import torch
import torchvision
from cfg import save_conv_bn, save_fc

def save_bottlenet_weights(model, fp):
    save_conv_bn(fp, model.conv1, model.bn1)
    save_conv_bn(fp, model.conv2, model.bn2)
    save_conv_bn(fp, model.conv3, model.bn3)
    if model.downsample:
        save_conv_bn(fp, model.downsample[0], model.downsample[1])

def save_resnet_weights(model, filename):
    fp = open(filename, 'wb')
    header = torch.IntTensor([0,0,0,0])
    header.numpy().tofile(fp)
    save_conv_bn(fp, model.conv1, model.bn1)
    for i in range(len(model.layer1._modules)):
        save_bottlenet_weights(model.layer1[i], fp)
    for i in range(len(model.layer2._modules)):
        save_bottlenet_weights(model.layer2[i], fp)
    for i in range(len(model.layer3._modules)):
        save_bottlenet_weights(model.layer3[i], fp)
    for i in range(len(model.layer4._modules)):
        save_bottlenet_weights(model.layer4[i], fp)
    save_fc(fp, model.fc)
    fp.close()


resnet50 = torchvision.models.resnet50(pretrained=True)
print('convert pytorch resnet50 to darkent, save resnet50.weights')
save_resnet_weights(resnet50, 'resnet50.weights')
