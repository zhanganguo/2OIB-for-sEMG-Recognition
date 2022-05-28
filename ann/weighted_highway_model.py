import torch


class WeightedHighwayFCN1(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.linear1 = torch.nn.Linear(in_features=input_shape, out_features=128, bias=False)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=128, bias=False)
        self.linear3 = torch.nn.Linear(in_features=128, out_features=256, bias=False)
        self.linear4 = torch.nn.Linear(in_features=256, out_features=num_classes, bias=False)
        self.Weighted = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.Weighted.data.fill_(1.0)

    def forward(self, x):
        x = torch.nn.ReLU()(self.linear1(x))
        r = torch.nn.ReLU()(self.linear2(x))
        x = self.Weighted * x + r
        x = torch.nn.ReLU()(self.linear3(x))
        x = self.linear4(x)

        return x


class WeightedHighwayFCN2(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.linear1 = torch.nn.Linear(in_features=input_shape, out_features=128, bias=False)
        self.linear2_0 = torch.nn.Linear(in_features=128, out_features=128, bias=False)
        self.linear2_1 = torch.nn.Linear(in_features=128, out_features=128, bias=False)
        self.linear3 = torch.nn.Linear(in_features=128, out_features=256, bias=False)
        self.linear4 = torch.nn.Linear(in_features=256, out_features=num_classes, bias=False)
        self.Weighted = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.Weighted.data.fill_(1.0)

    def forward(self, x):
        x = torch.nn.ReLU()(self.linear1(x))
        r = torch.nn.ReLU()(self.linear2_0(x))
        r = torch.nn.ReLU()(self.linear2_1(r))
        x = r + self.Weighted * x
        x = torch.nn.ReLU()(self.linear3(x))
        x = self.linear4(x)

        return x


class WeightedHighwayCNN1(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.conv1 = torch.nn.Conv2d(in_channels=input_shape[-1], out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=0, bias=False)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=0, bias=False)
        self.linear1 = torch.nn.Linear(in_features=64*4*4, out_features=128, bias=False)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=num_classes, bias=False)

        self.Weighted = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.Weighted.data.fill_(1.0)

    def forward(self, x):
        x = torch.nn.ReLU()(self.conv1(x))
        r = torch.nn.ReLU()(self.conv2(x))
        x = self.Weighted * x + r
        x = torch.nn.ReLU()(self.conv3(x))
        x = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x)
        x = torch.nn.ReLU()(self.conv4(x))
        x = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x)
        x = x.view(-1, self.num_flat_features(x))
        x = torch.nn.ReLU()(self.linear1(x))
        x = self.linear2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class WeightedHighwayCNN2(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.conv1 = torch.nn.Conv2d(in_channels=input_shape[-1], out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2_0 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2_1 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=0, bias=False)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=0, bias=False)
        self.linear1 = torch.nn.Linear(in_features=64*4*4, out_features=128, bias=False)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=num_classes, bias=False)

        self.Weighted = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.Weighted.data.fill_(1.0)

    def forward(self, x):
        x = torch.nn.ReLU()(self.conv1(x))
        r = torch.nn.ReLU()(self.conv2_0(x))
        r = torch.nn.ReLU()(self.conv2_1(r))
        x = self.Weighted * x + r
        x = torch.nn.ReLU()(self.conv3(x))
        x = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x)
        x = torch.nn.ReLU()(self.conv4(x))
        x = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x)
        x = x.view(-1, self.num_flat_features(x))
        x = torch.nn.ReLU()(self.linear1(x))
        x = self.linear2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class WeightedHighwayCNN3(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.conv1 = torch.nn.Conv2d(in_channels=input_shape[-1], out_channels=64, kernel_size=(3, 3), padding=1,
                                     bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False)
        self.conv4_0 = torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(3, 3), padding=1, bias=False)
        self.conv4_1 = torch.nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(3, 3), padding=1, bias=False)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False)
        self.conv6 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, bias=False)
        self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False)
        self.linear1 = torch.nn.Linear(in_features=512 * 4 * 4, out_features=256, bias=False)
        self.linear2 = torch.nn.Linear(in_features=256, out_features=num_classes, bias=False)

        self.Weighted1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.Weighted1.data.fill_(1.0)
        self.Weighted2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.Weighted2.data.fill_(1.0)
        self.Weighted3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.Weighted3.data.fill_(1.0)

    def forward(self, x):
        x = torch.nn.ReLU()(self.conv1(x))
        r = torch.nn.ReLU()(self.conv2(x))
        x = self.Weighted1 * x + r
        x = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x)
        x = torch.nn.ReLU()(self.conv3(x))
        r = torch.nn.ReLU()(self.conv4_0(x))
        r = torch.nn.ReLU()(self.conv4_1(r))
        x = self.Weighted2 * x + r
        x = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x)
        x = torch.nn.ReLU()(self.conv5(x))
        r = torch.nn.ReLU()(self.conv6(x))
        x = self.Weighted3 * x + r
        x = torch.nn.ReLU()(self.conv7(x))
        x = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x)
        x = x.view(-1, self.num_flat_features(x))
        x = torch.nn.ReLU()(self.linear1(x))
        x = self.linear2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class WeightedHighwayCNN_18(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        super(WeightedHighwayCNN_18, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv1 = torch.nn.Conv2d(input_shape[-1], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.conv1.out_channels)

        self.conv1_1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = torch.nn.BatchNorm2d(self.conv1_1.out_channels)
        self.conv1_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = torch.nn.BatchNorm2d(self.conv1_2.out_channels)
        self.conv1_3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_3 = torch.nn.BatchNorm2d(self.conv1_3.out_channels)

        self.conv2_1 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2_1 = torch.nn.BatchNorm2d(self.conv2_1.out_channels)
        self.conv2_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                                       bias=False)
        self.bn2_2 = torch.nn.BatchNorm2d(self.conv2_2.out_channels)
        self.conv2_3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, bias=False)
        self.bn2_3 = torch.nn.BatchNorm2d(self.conv2_3.out_channels)
        self.conv2_4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                                       bias=False)
        self.bn2_4 = torch.nn.BatchNorm2d(self.conv2_4.out_channels)
        self.conv2_5 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                                       bias=False)
        self.bn2_5 = torch.nn.BatchNorm2d(self.conv2_5.out_channels)

        self.conv3_1 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1,
                                       bias=False)
        self.bn3_1 = torch.nn.BatchNorm2d(self.conv3_1.out_channels)
        self.conv3_2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1,
                                       bias=False)
        self.bn3_2 = torch.nn.BatchNorm2d(self.conv3_2.out_channels)
        self.conv3_3 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False)
        self.bn3_3 = torch.nn.BatchNorm2d(self.conv3_3.out_channels)
        self.conv3_4 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1,
                                       bias=False)
        self.bn3_4 = torch.nn.BatchNorm2d(self.conv3_4.out_channels)
        self.conv3_5 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1,
                                       bias=False)
        self.bn3_5 = torch.nn.BatchNorm2d(self.conv3_5.out_channels)

        self.conv4_1 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1,
                                       bias=False)
        self.bn4_1 = torch.nn.BatchNorm2d(self.conv4_1.out_channels)
        self.conv4_2 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1,
                                       bias=False)
        self.bn4_2 = torch.nn.BatchNorm2d(self.conv4_2.out_channels)
        self.conv4_3 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, bias=False)
        self.bn4_3 = torch.nn.BatchNorm2d(self.conv4_3.out_channels)
        self.conv4_4 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1,
                                       bias=False)
        self.bn4_4 = torch.nn.BatchNorm2d(self.conv4_4.out_channels)
        self.conv4_5 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1,
                                       bias=False)
        self.bn4_5 = torch.nn.BatchNorm2d(self.conv4_5.out_channels)

        self.fc = torch.nn.Linear(512, num_classes, bias=False)

        self.Weighted1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.Weighted1.data.fill_(1.0)
        self.Weighted2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.Weighted2.data.fill_(1.0)
        self.Weighted3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.Weighted3.data.fill_(1.0)
        self.Weighted4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.Weighted4.data.fill_(1.0)

    def forward(self, x):
        x = torch.nn.ReLU()(self.bn1(self.conv1(x)))

        x = torch.nn.ReLU()(self.bn1_1(self.conv1_1(x)))
        x = torch.nn.ReLU()(self.bn1_2(self.conv1_2(x)))
        out = torch.nn.ReLU()(self.bn1_3(self.conv1_3(x)))
        x = out + x * self.Weighted1

        x = torch.nn.ReLU()(self.bn2_1(self.conv2_1(x)))
        x = torch.nn.ReLU()(self.bn2_2(self.conv2_2(x)))
        out = torch.nn.ReLU()(self.bn2_3(self.conv2_3(x)))
        x = out + x * self.Weighted2
        x = torch.nn.ReLU()(self.bn2_4(self.conv2_4(x)))
        x = torch.nn.ReLU()(self.bn2_5(self.conv2_5(x)))

        x = torch.nn.ReLU()(self.bn3_1(self.conv3_1(x)))
        x = torch.nn.ReLU()(self.bn3_2(self.conv3_2(x)))
        out = torch.nn.ReLU()(self.bn3_3(self.conv3_3(x)))
        x = out + x * self.Weighted3
        x = torch.nn.ReLU()(self.bn3_4(self.conv3_4(x)))
        x = torch.nn.ReLU()(self.bn3_5(self.conv3_5(x)))

        x = torch.nn.ReLU()(self.bn4_1(self.conv4_1(x)))
        x = torch.nn.ReLU()(self.bn4_2(self.conv4_2(x)))
        out = torch.nn.ReLU()(self.bn4_3(self.conv4_3(x)))
        x = out + x * self.Weighted4
        x = torch.nn.ReLU()(self.bn4_4(self.conv4_4(x)))
        x = torch.nn.ReLU()(self.bn4_5(self.conv4_5(x)))

        x = torch.nn.functional.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
