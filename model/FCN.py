import torch.nn as nn
import torchvision.models as models

class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int, kernel:int, pad:int=1):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, kernel, padding=pad)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class FCN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, **_):
        super(FCN, self).__init__()
        #self.relu = nn.ReLU(inplace=True)

        # VGG
        vgg = models.vgg16(pretrained=pretrained)
        encoder = list(vgg.features.children())

        self.pool3 = nn.Sequential(*encoder[0:17])

        self.pool4 = nn.Sequential(*encoder[17:24])

        self.pool5 = nn.Sequential(*encoder[24:31])

        self.conv_6 = nn.Sequential(nn.Conv2d(512, 4096, (7, 7), padding=1),nn.ReLU(inplace=True))

        self.conv_7 = nn.Sequential(nn.Conv2d(4096, 4096, (1, 1)),nn.ReLU(inplace=True))

        self.conv_8 = nn.Conv2d(4096, num_classes, (1, 1))

        self.convTrans_9 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.convTrans_10 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1x1_3_4 = nn.Conv2d(256, num_classes, (1, 1))

        self.conv1x1_4_4 = nn.Conv2d(512, num_classes, (1, 1))

        self.drop6 = nn.Dropout2d()

        self.drop7 = nn.Dropout2d()

        self.convTrans_11 = nn.UpsamplingBilinear2d(scale_factor=8)

        #self.final = nn.Sequential(nn.Conv2d(num_classes, num_classes, (1, 1)),nn.Softmax2d())

    def forward(self, x):
        x = x.float()
        x = self.pool3(x)
        f3 = self.conv1x1_3_4(x)
        x = self.pool4(x)
        f4 = self.conv1x1_4_4(x)
        x = self.pool5(x)
        x = self.conv_6(x)
        x = self.drop6(x)
        x = self.conv_7(x)
        x = self.drop7(x)
        x = self.conv_8(x)
        x = self.convTrans_9(x)
        x += f4
        x = self.convTrans_10(x)
        x += f3
        x = self.convTrans_11(x)

        output = x
        return output
