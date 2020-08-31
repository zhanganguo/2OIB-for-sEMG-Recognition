import torch


class FCN(torch.nn.Module):
    def __init__(self, input_shape, num_classes, K=256, logvar_t=-1.0, train_logvar_t=False):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.scale_factor = 2

        self.conv1 = torch.nn.Conv2d(in_channels=input_shape[0], out_channels=32 * self.scale_factor, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=64 * self.scale_factor, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=64 * self.scale_factor, kernel_size=(3, 3), stride=(2, 2), padding=2, dilation=2, bias=False)
        self.conv4 = torch.nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=128 * self.scale_factor, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv5 = torch.nn.Conv2d(in_channels=self.conv4.out_channels, out_channels=128 * self.scale_factor, kernel_size=(3, 3), stride=(2, 2), padding=2, dilation=2, bias=False)
        self.conv6 = torch.nn.Conv2d(in_channels=self.conv5.out_channels, out_channels=128 * self.scale_factor,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        # self.linear1 = torch.nn.Linear(in_features=self.conv5.out_channels*input_shape[0]//4*input_shape[1]//4, out_features=128, bias=False)
        # self.gap = torch.nn.AvgPool2d(kernel_size=(3, 3))
        self.linear_0 = torch.nn.Linear(in_features=self.conv6.out_channels*3*3, out_features=K, bias=False)
        self.linear = torch.nn.Linear(in_features=K, out_features=num_classes, bias=False)

        if train_logvar_t:
            self.logvar_t = torch.nn.Parameter(torch.Tensor([logvar_t]))
        else:
            self.logvar_t = torch.Tensor([logvar_t])

        if torch.cuda.is_available():
            self.logvar_t = self.logvar_t.cuda()

    def encode_features(self, x, random=True):
        x = torch.nn.ReLU()(self.conv1(x))
        x = torch.nn.ReLU()(self.conv2(x))
        x = torch.nn.ReLU()(self.conv3(x))
        x = torch.nn.ReLU()(self.conv4(x))
        x = torch.nn.ReLU()(self.conv5(x))
        x = torch.nn.ReLU()(self.conv6(x))
        mean_t = x.view(-1, self.num_flat_features(x))

        if random:
            t = mean_t + self.add_noise(mean_t)
        else:
            t = mean_t
        return t

    def decode_features(self, t):
        x = torch.nn.Dropout(0.4)(t)
        x = self.linear_0(x)
        t = self.linear(x)
        return t

    def add_noise(self, mean_t):
        noise = torch.exp(0.5 * self.logvar_t).cuda() * torch.randn_like(mean_t)

        return noise

    def forward(self, x):
        # x = torch.nn.ReLU()(self.conv1(x))
        # x = torch.nn.ReLU()(self.conv2(x))
        # x = torch.nn.ReLU()(self.conv3(x))
        # x = torch.nn.ReLU()(self.conv4(x))
        # x = torch.nn.ReLU()(self.conv5(x))
        # x = x.view(-1, self.num_flat_features(x))
        # x = torch.nn.Dropout(0.4)(x)
        # x = self.linear_0(x)
        # x = self.linear(x)
        t = self.encode_features(x)
        logits_y = self.decode_features(t)

        return logits_y

    def num_flat_features(self, x):
        size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class FCN2(torch.nn.Module):
    def __init__(self, input_shape, num_classes, K=256, logvar_t=-1.0, train_logvar_t=False):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.scale_factor = 3

        self.conv1 = torch.nn.Conv2d(in_channels=input_shape[0], out_channels=32 * self.scale_factor, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=64 * self.scale_factor, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=128 * self.scale_factor, kernel_size=(3, 3), stride=(2, 2), padding=2, dilation=2, bias=False)
        self.conv4_1 = torch.nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=128 * self.scale_factor, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv4_2 = torch.nn.Conv2d(in_channels=self.conv4_1.out_channels, out_channels=128 * self.scale_factor,
                                       kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv5 = torch.nn.Conv2d(in_channels=self.conv4_2.out_channels, out_channels=128 * self.scale_factor, kernel_size=(3, 3), stride=(2, 2), padding=2, dilation=2, bias=False)
        self.conv6 = torch.nn.Conv2d(in_channels=self.conv5.out_channels, out_channels=128 * self.scale_factor,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        # self.linear1 = torch.nn.Linear(in_features=self.conv5.out_channels*input_shape[0]//4*input_shape[1]//4, out_features=128, bias=False)
        # self.gap = torch.nn.AvgPool2d(kernel_size=(3, 3))
        self.linear_0 = torch.nn.Linear(in_features=self.conv6.out_channels*3*3, out_features=K, bias=False)
        self.linear = torch.nn.Linear(in_features=K, out_features=num_classes, bias=False)

        if train_logvar_t:
            self.logvar_t = torch.nn.Parameter(torch.Tensor([logvar_t]))
        else:
            self.logvar_t = torch.Tensor([logvar_t])

        if torch.cuda.is_available():
            self.logvar_t = self.logvar_t.cuda()

    def encode_features(self, x, random=True):
        x = torch.nn.ReLU()(self.conv1(x))
        x = torch.nn.ReLU()(self.conv2(x))
        x = torch.nn.ReLU()(self.conv3(x))
        b = torch.nn.ReLU()(self.conv4_1(x))
        b = torch.nn.ReLU()(self.conv4_2(b))
        x = x + b
        x = torch.nn.ReLU()(self.conv5(x))
        x = torch.nn.ReLU()(self.conv6(x))
        mean_t = x.view(-1, self.num_flat_features(x))

        if random:
            t = mean_t + self.add_noise(mean_t)
        else:
            t = mean_t
        return t

    def decode_features(self, t):
        x = torch.nn.Dropout(0.4)(t)
        x = self.linear_0(x)
        t = self.linear(x)
        return t

    def add_noise(self, mean_t):
        noise = torch.exp(0.5 * self.logvar_t).cuda() * torch.randn_like(mean_t)

        return noise

    def forward(self, x):
        # x = torch.nn.ReLU()(self.conv1(x))
        # x = torch.nn.ReLU()(self.conv2(x))
        # x = torch.nn.ReLU()(self.conv3(x))
        # x = torch.nn.ReLU()(self.conv4(x))
        # x = torch.nn.ReLU()(self.conv5(x))
        # x = x.view(-1, self.num_flat_features(x))
        # x = torch.nn.Dropout(0.4)(x)
        # x = self.linear_0(x)
        # x = self.linear(x)
        t = self.encode_features(x)
        logits_y = self.decode_features(t)

        return logits_y

    def num_flat_features(self, x):
        size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class FCN1_DB2(torch.nn.Module):
    def __init__(self, input_shape, num_classes, K=256, logvar_t=-1.0, train_logvar_t=False):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.scale_factor = 2

        self.conv1 = torch.nn.Conv2d(in_channels=input_shape[0], out_channels=32 * self.scale_factor, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=64 * self.scale_factor, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=64 * self.scale_factor, kernel_size=(3, 3), stride=(2, 2), padding=2, dilation=2, bias=False)
        self.conv4 = torch.nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=128 * self.scale_factor, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv5 = torch.nn.Conv2d(in_channels=self.conv4.out_channels, out_channels=128 * self.scale_factor, kernel_size=(3, 3), stride=(2, 2), padding=2, dilation=2, bias=False)
        self.conv6 = torch.nn.Conv2d(in_channels=self.conv5.out_channels, out_channels=128 * self.scale_factor,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        # self.linear1 = torch.nn.Linear(in_features=self.conv5.out_channels*input_shape[0]//4*input_shape[1]//4, out_features=128, bias=False)
        # self.gap = torch.nn.AvgPool2d(kernel_size=(3, 3))
        self.linear_0 = torch.nn.Linear(in_features=self.conv6.out_channels*4*3, out_features=K, bias=False)
        self.linear = torch.nn.Linear(in_features=K, out_features=num_classes, bias=False)

        if train_logvar_t:
            self.logvar_t = torch.nn.Parameter(torch.Tensor([logvar_t]))
        else:
            self.logvar_t = torch.Tensor([logvar_t])

        if torch.cuda.is_available():
            self.logvar_t = self.logvar_t.cuda()

    def encode_features(self, x, random=True):
        x = torch.nn.ReLU()(self.conv1(x))
        x = torch.nn.ReLU()(self.conv2(x))
        x = torch.nn.ReLU()(self.conv3(x))
        x = torch.nn.ReLU()(self.conv4(x))
        x = torch.nn.ReLU()(self.conv5(x))
        x = torch.nn.ReLU()(self.conv6(x))
        mean_t = x.view(-1, self.num_flat_features(x))

        if random:
            t = mean_t + self.add_noise(mean_t)
        else:
            t = mean_t
        return t

    def decode_features(self, t):
        x = torch.nn.Dropout(0.4)(t)
        x = self.linear_0(x)
        t = self.linear(x)
        return t

    def add_noise(self, mean_t):
        noise = torch.exp(0.5 * self.logvar_t).cuda() * torch.randn_like(mean_t)

        return noise

    def forward(self, x):
        # x = torch.nn.ReLU()(self.conv1(x))
        # x = torch.nn.ReLU()(self.conv2(x))
        # x = torch.nn.ReLU()(self.conv3(x))
        # x = torch.nn.ReLU()(self.conv4(x))
        # x = torch.nn.ReLU()(self.conv5(x))
        # x = x.view(-1, self.num_flat_features(x))
        # x = torch.nn.Dropout(0.4)(x)
        # x = self.linear_0(x)
        # x = self.linear(x)
        t = self.encode_features(x)
        logits_y = self.decode_features(t)

        return logits_y

    def num_flat_features(self, x):
        size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class FCN2_DB2(torch.nn.Module):
    def __init__(self, input_shape, num_classes, K=256, logvar_t=-1.0, train_logvar_t=False):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.scale_factor = 3

        self.conv1 = torch.nn.Conv2d(in_channels=input_shape[0], out_channels=32 * self.scale_factor, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=64 * self.scale_factor, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=128 * self.scale_factor, kernel_size=(3, 3), stride=(2, 2), padding=2, dilation=2, bias=False)
        self.conv4_1 = torch.nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=128 * self.scale_factor, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv4_2 = torch.nn.Conv2d(in_channels=self.conv4_1.out_channels, out_channels=128 * self.scale_factor,
                                       kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv5 = torch.nn.Conv2d(in_channels=self.conv4_2.out_channels, out_channels=128 * self.scale_factor, kernel_size=(3, 3), stride=(2, 2), padding=2, dilation=2, bias=False)
        self.conv6 = torch.nn.Conv2d(in_channels=self.conv5.out_channels, out_channels=128 * self.scale_factor,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        # self.linear1 = torch.nn.Linear(in_features=self.conv5.out_channels*input_shape[0]//4*input_shape[1]//4, out_features=128, bias=False)
        # self.gap = torch.nn.AvgPool2d(kernel_size=(3, 3))
        self.linear_0 = torch.nn.Linear(in_features=self.conv6.out_channels*4*3, out_features=K, bias=False)
        self.linear = torch.nn.Linear(in_features=K, out_features=num_classes, bias=False)

        if train_logvar_t:
            self.logvar_t = torch.nn.Parameter(torch.Tensor([logvar_t]))
        else:
            self.logvar_t = torch.Tensor([logvar_t])

        if torch.cuda.is_available():
            self.logvar_t = self.logvar_t.cuda()

    def encode_features(self, x, random=True):
        x = torch.nn.ReLU()(self.conv1(x))
        x = torch.nn.ReLU()(self.conv2(x))
        x = torch.nn.ReLU()(self.conv3(x))
        b = torch.nn.ReLU()(self.conv4_1(x))
        b = torch.nn.ReLU()(self.conv4_2(b))
        x = x + b
        x = torch.nn.ReLU()(self.conv5(x))
        x = torch.nn.ReLU()(self.conv6(x))
        mean_t = x.view(-1, self.num_flat_features(x))

        if random:
            t = mean_t + self.add_noise(mean_t)
        else:
            t = mean_t
        return t

    def decode_features(self, t):
        x = torch.nn.Dropout(0.4)(t)
        x = self.linear_0(x)
        t = self.linear(x)
        return t

    def add_noise(self, mean_t):
        noise = torch.exp(0.5 * self.logvar_t).cuda() * torch.randn_like(mean_t)

        return noise

    def forward(self, x):
        # x = torch.nn.ReLU()(self.conv1(x))
        # x = torch.nn.ReLU()(self.conv2(x))
        # x = torch.nn.ReLU()(self.conv3(x))
        # x = torch.nn.ReLU()(self.conv4(x))
        # x = torch.nn.ReLU()(self.conv5(x))
        # x = x.view(-1, self.num_flat_features(x))
        # x = torch.nn.Dropout(0.4)(x)
        # x = self.linear_0(x)
        # x = self.linear(x)
        t = self.encode_features(x)
        logits_y = self.decode_features(t)

        return logits_y

    def num_flat_features(self, x):
        size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
