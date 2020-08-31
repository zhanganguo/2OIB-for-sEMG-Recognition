import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


class FCN(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.logvar_t = 0

        self.scale_factor = 2

        self.conv1 = torch.nn.Conv2d(in_channels=input_shape[0], out_channels=32 * self.scale_factor, kernel_size=(3, 3), stride=(1, 1),
                                     padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=64 * self.scale_factor,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=64 * self.scale_factor,
                                     kernel_size=(3, 3), stride=(2, 2), padding=2, dilation=2, bias=False)
        self.conv4 = torch.nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=128 * self.scale_factor,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv5 = torch.nn.Conv2d(in_channels=self.conv4.out_channels, out_channels=128 * self.scale_factor,
                                     kernel_size=(3, 3), stride=(2, 2), padding=2, dilation=2, bias=False)
        self.conv6 = torch.nn.Conv2d(in_channels=self.conv5.out_channels, out_channels=128 * self.scale_factor,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        # self.linear1 = torch.nn.Linear(in_features=self.conv5.out_channels*input_shape[0]//4*input_shape[1]//4, out_features=128, bias=False)
        # self.gap = torch.nn.AvgPool2d(kernel_size=(3, 3))
        self.linear_0 = torch.nn.Linear(in_features=self.conv6.out_channels * 3 * 3,
                                        out_features=self.conv6.out_channels * self.scale_factor, bias=False)
        self.linear = torch.nn.Linear(in_features=self.conv6.out_channels * self.scale_factor, out_features=num_classes,
                                      bias=False)

    def forward(self, x):
        x = torch.nn.ReLU()(self.conv1(x))
        x = torch.nn.ReLU()(self.conv2(x))
        x = torch.nn.ReLU()(self.conv3(x))
        x = torch.nn.ReLU()(self.conv4(x))
        x = torch.nn.ReLU()(self.conv5(x))
        x = torch.nn.ReLU()(self.conv6(x))
        x = x.view(-1, self.num_flat_features(x))
        x = torch.nn.Dropout(0.3)(x)
        x = self.linear_0(x)
        x = self.linear(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class FCN2(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.logvar_t = 0

        self.scale_factor = 2

        self.conv1 = torch.nn.Conv2d(in_channels=input_shape[0], out_channels=32 * self.scale_factor, kernel_size=(3, 3), stride=(1, 1),
                                     padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=64 * self.scale_factor,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=128 * self.scale_factor,
                                     kernel_size=(3, 3), stride=(2, 2), padding=2, dilation=2, bias=False)
        self.conv4_1 = torch.nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=128 * self.scale_factor,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv4_2 = torch.nn.Conv2d(in_channels=self.conv4_1.out_channels, out_channels=128 * self.scale_factor,
                                       kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv5 = torch.nn.Conv2d(in_channels=self.conv4_2.out_channels, out_channels=128 * self.scale_factor,
                                     kernel_size=(3, 3), stride=(2, 2), padding=2, dilation=2, bias=False)
        self.conv6 = torch.nn.Conv2d(in_channels=self.conv5.out_channels, out_channels=128 * self.scale_factor,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        # self.linear1 = torch.nn.Linear(in_features=self.conv5.out_channels*input_shape[0]//4*input_shape[1]//4, out_features=128, bias=False)
        # self.gap = torch.nn.AvgPool2d(kernel_size=(3, 3))
        self.linear_0 = torch.nn.Linear(in_features=self.conv6.out_channels * 3 * 3,
                                        out_features=self.conv6.out_channels * self.scale_factor, bias=False)
        self.linear = torch.nn.Linear(in_features=self.conv6.out_channels * self.scale_factor, out_features=num_classes,
                                      bias=False)

    def forward(self, x):
        x = torch.nn.ReLU()(self.conv1(x))
        x = torch.nn.ReLU()(self.conv2(x))
        x = torch.nn.ReLU()(self.conv3(x))
        b = torch.nn.ReLU()(self.conv4_1(x))
        b = torch.nn.ReLU()(self.conv4_2(b))
        x = x + b
        x = torch.nn.ReLU()(self.conv5(x))
        x = torch.nn.ReLU()(self.conv6(x))
        x = x.view(-1, self.num_flat_features(x))
        x = torch.nn.Dropout(0.3)(x)
        x = self.linear_0(x)
        x = self.linear(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
