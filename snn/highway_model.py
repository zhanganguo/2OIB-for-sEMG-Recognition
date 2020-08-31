import torch

from ann.highway_model import CNN1, HighwayCNN1, CNN2, HighwayCNN2, FCN1, HighwayFCN1, FCN2, HighwayFCN2, CNN3, \
    HighwayCNN3, HighwayCNN_18, CNN_18
from snn.spiking_operation import SpikingFeedForwardCell
from snn.utils import threshold


class SpikingCNN1(CNN1):
    def __init__(self, input_shape, num_classes, spiking_neuron, if_param, device, seq_length, max_firing_rate, dt):
        super(SpikingCNN1, self).__init__(input_shape=input_shape, num_classes=num_classes)
        self.spiking_conv1 = SpikingFeedForwardCell(spiking_op=self.conv1, spiking_neuron=spiking_neuron(
            shape=(32, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv2 = SpikingFeedForwardCell(spiking_op=self.conv2, spiking_neuron=spiking_neuron(
            shape=(32, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv3 = SpikingFeedForwardCell(spiking_op=self.conv3, spiking_neuron=spiking_neuron(
            shape=(32, self.input_shape[0]-4, self.input_shape[1]-4), if_params=if_param))
        self.spiking_conv4 = SpikingFeedForwardCell(spiking_op=self.conv4, spiking_neuron=spiking_neuron(
            shape=(64, (self.input_shape[0]-4)//2-4, (self.input_shape[1]-4)//2-4), if_params=if_param))
        self.spiking_linear1 = SpikingFeedForwardCell(spiking_op=self.linear1, spiking_neuron=spiking_neuron(shape=(128,), if_params=if_param))
        self.spiking_linear2 = SpikingFeedForwardCell(spiking_op=self.linear2, spiking_neuron=spiking_neuron(shape=(num_classes,), if_params=if_param))

        self.seq_length = seq_length
        self.max_firing_rate = max_firing_rate
        self.dt = dt
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        self.spiking_conv1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear2.initial_state(batch_size=batch_size, device=self.device)
        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        output_record = torch.zeros((self.seq_length, batch_size), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            inp_image = threshold(x, spike_snapshot)

            x_spike = self.spiking_conv1.forward(inp_image)
            x_spike = self.spiking_conv2.forward(x_spike)
            x_spike = self.spiking_conv3.forward(x_spike)
            x_spike = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x_spike)
            x_spike = self.spiking_conv4.forward(x_spike)
            x_spike = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x_spike)
            x_spike = x_spike.view(-1, self.num_flat_features(x_spike))
            x_spike = self.spiking_linear1.forward(x_spike)
            x_spike = self.spiking_linear2.forward(x_spike)

            spike_out += x_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            output_record[i, :] = torch.transpose(pred, 1, 0)

        return spike_out, output_record


class SpikingHighwayCNN1(HighwayCNN1):
    def __init__(self, input_shape, num_classes, spiking_neuron, if_param, device, seq_length, max_firing_rate, dt):
        super(SpikingHighwayCNN1, self).__init__(input_shape=input_shape, num_classes=num_classes)
        self.spiking_conv1 = SpikingFeedForwardCell(spiking_op=self.conv1, spiking_neuron=spiking_neuron(
            shape=(32, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv2 = SpikingFeedForwardCell(spiking_op=self.conv2, spiking_neuron=spiking_neuron(
            shape=(32, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv3 = SpikingFeedForwardCell(spiking_op=self.conv3, spiking_neuron=spiking_neuron(
            shape=(32, self.input_shape[0]-4, self.input_shape[1]-4), if_params=if_param))
        self.spiking_conv4 = SpikingFeedForwardCell(spiking_op=self.conv4, spiking_neuron=spiking_neuron(
            shape=(64, (self.input_shape[0]-4)//2-4, (self.input_shape[1]-4)//2-4), if_params=if_param))
        self.spiking_linear1 = SpikingFeedForwardCell(spiking_op=self.linear1, spiking_neuron=spiking_neuron(shape=(128,), if_params=if_param))
        self.spiking_linear2 = SpikingFeedForwardCell(spiking_op=self.linear2, spiking_neuron=spiking_neuron(shape=(num_classes,), if_params=if_param))

        self.seq_length = seq_length
        self.max_firing_rate = max_firing_rate
        self.dt = dt
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        self.spiking_conv1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear2.initial_state(batch_size=batch_size, device=self.device)
        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        output_record = torch.zeros((self.seq_length, batch_size), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            inp_image = threshold(x, spike_snapshot)

            x_spike = self.spiking_conv1.forward(inp_image)
            r_spike = self.spiking_conv2.forward(x_spike)
            x_spike = x_spike + r_spike
            x_spike = threshold(x_spike, 1)
            x_spike = self.spiking_conv3.forward(x_spike)
            x_spike = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x_spike)
            x_spike = self.spiking_conv4.forward(x_spike)
            x_spike = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x_spike)
            x_spike = x_spike.view(-1, self.num_flat_features(x_spike))
            x_spike = self.spiking_linear1.forward(x_spike)
            x_spike = self.spiking_linear2.forward(x_spike)

            spike_out += x_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            output_record[i, :] = torch.transpose(pred, 1, 0)

        return spike_out, output_record


class SpikingCNN2(CNN2):
    def __init__(self, input_shape, num_classes, spiking_neuron, if_param, device, seq_length, max_firing_rate, dt):
        super(SpikingCNN2, self).__init__(input_shape=input_shape, num_classes=num_classes)
        self.spiking_conv1 = SpikingFeedForwardCell(spiking_op=self.conv1, spiking_neuron=spiking_neuron(
            shape=(32, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv2_0 = SpikingFeedForwardCell(spiking_op=self.conv2_0, spiking_neuron=spiking_neuron(
            shape=(32, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv2_1 = SpikingFeedForwardCell(spiking_op=self.conv2_1, spiking_neuron=spiking_neuron(
            shape=(32, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv3 = SpikingFeedForwardCell(spiking_op=self.conv3, spiking_neuron=spiking_neuron(
            shape=(32, self.input_shape[0]-4, self.input_shape[1]-4), if_params=if_param))
        self.spiking_conv4 = SpikingFeedForwardCell(spiking_op=self.conv4, spiking_neuron=spiking_neuron(
            shape=(64, (self.input_shape[0]-4)//2-4, (self.input_shape[1]-4)//2-4), if_params=if_param))
        self.spiking_linear1 = SpikingFeedForwardCell(spiking_op=self.linear1, spiking_neuron=spiking_neuron(shape=(128,), if_params=if_param))
        self.spiking_linear2 = SpikingFeedForwardCell(spiking_op=self.linear2, spiking_neuron=spiking_neuron(shape=(num_classes,), if_params=if_param))

        self.seq_length = seq_length
        self.max_firing_rate = max_firing_rate
        self.dt = dt
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        self.spiking_conv1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2_0.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2_1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear2.initial_state(batch_size=batch_size, device=self.device)
        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        output_record = torch.zeros((self.seq_length, batch_size), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            inp_image = threshold(x, spike_snapshot)

            x_spike = self.spiking_conv1.forward(inp_image)
            x_spike = self.spiking_conv2_0.forward(x_spike)
            x_spike = self.spiking_conv2_1.forward(x_spike)
            x_spike = self.spiking_conv3.forward(x_spike)
            x_spike = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x_spike)
            x_spike = self.spiking_conv4.forward(x_spike)
            x_spike = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x_spike)
            x_spike = x_spike.view(-1, self.num_flat_features(x_spike))
            x_spike = self.spiking_linear1.forward(x_spike)
            x_spike = self.spiking_linear2.forward(x_spike)

            spike_out += x_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            output_record[i, :] = torch.transpose(pred, 1, 0)

        return spike_out, output_record


class SpikingHighwayCNN2(HighwayCNN2):
    def __init__(self, input_shape, num_classes, spiking_neuron, if_param, device, seq_length, max_firing_rate, dt):
        super(SpikingHighwayCNN2, self).__init__(input_shape=input_shape, num_classes=num_classes)
        self.spiking_conv1 = SpikingFeedForwardCell(spiking_op=self.conv1, spiking_neuron=spiking_neuron(
            shape=(32, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv2_0 = SpikingFeedForwardCell(spiking_op=self.conv2_0, spiking_neuron=spiking_neuron(
            shape=(32, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv2_1 = SpikingFeedForwardCell(spiking_op=self.conv2_1, spiking_neuron=spiking_neuron(
            shape=(32, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv3 = SpikingFeedForwardCell(spiking_op=self.conv3, spiking_neuron=spiking_neuron(
            shape=(32, self.input_shape[0]-4, self.input_shape[1]-4), if_params=if_param))
        self.spiking_conv4 = SpikingFeedForwardCell(spiking_op=self.conv4, spiking_neuron=spiking_neuron(
            shape=(64, (self.input_shape[0]-4)//2-4, (self.input_shape[1]-4)//2-4), if_params=if_param))
        self.spiking_linear1 = SpikingFeedForwardCell(spiking_op=self.linear1, spiking_neuron=spiking_neuron(shape=(128,), if_params=if_param))
        self.spiking_linear2 = SpikingFeedForwardCell(spiking_op=self.linear2, spiking_neuron=spiking_neuron(shape=(num_classes,), if_params=if_param))

        self.seq_length = seq_length
        self.max_firing_rate = max_firing_rate
        self.dt = dt
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        self.spiking_conv1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2_0.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2_1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear2.initial_state(batch_size=batch_size, device=self.device)
        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        output_record = torch.zeros((self.seq_length, batch_size), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            inp_image = threshold(x, spike_snapshot)

            x_spike = self.spiking_conv1.forward(inp_image)
            r_spike = self.spiking_conv2_0.forward(x_spike)
            r_spike = self.spiking_conv2_1.forward(r_spike)
            x_spike = x_spike + r_spike
            # x_spike = threshold(x_spike, 1)
            x_spike = self.spiking_conv3.forward(x_spike)
            x_spike = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x_spike)
            x_spike = self.spiking_conv4.forward(x_spike)
            x_spike = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x_spike)
            x_spike = x_spike.view(-1, self.num_flat_features(x_spike))
            x_spike = self.spiking_linear1.forward(x_spike)
            x_spike = self.spiking_linear2.forward(x_spike)

            spike_out += x_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            output_record[i, :] = torch.transpose(pred, 1, 0)
            # cur_correct = pred.eq(self.y.data.view_as(pred)).cpu().sum()

        return spike_out, output_record


class SpikingFCN1(FCN1):
    def __init__(self, input_shape, num_classes, spiking_neuron, if_param, device, seq_length, max_firing_rate, dt):
        super(SpikingFCN1, self).__init__(input_shape=input_shape, num_classes=num_classes)
        self.spiking_linear1 = SpikingFeedForwardCell(spiking_op=self.linear1, spiking_neuron=spiking_neuron(
            shape=(128, ), if_params=if_param))
        self.spiking_linear2 = SpikingFeedForwardCell(spiking_op=self.linear2, spiking_neuron=spiking_neuron(
            shape=(128, ), if_params=if_param))
        self.spiking_linear3 = SpikingFeedForwardCell(spiking_op=self.linear3, spiking_neuron=spiking_neuron(
            shape=(256, ), if_params=if_param))
        self.spiking_linear4 = SpikingFeedForwardCell(spiking_op=self.linear4, spiking_neuron=spiking_neuron(
            shape=(num_classes,), if_params=if_param))

        self.seq_length = seq_length
        self.max_firing_rate = max_firing_rate
        self.dt = dt
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        self.spiking_linear3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear4.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear2.initial_state(batch_size=batch_size, device=self.device)
        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        output_record = torch.zeros((self.seq_length, batch_size), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            inp_image = threshold(x, spike_snapshot)

            x_spike = self.spiking_linear1.forward(inp_image)
            x_spike = self.spiking_linear2.forward(x_spike)
            x_spike = self.spiking_linear3.forward(x_spike)
            x_spike = self.spiking_linear4.forward(x_spike)
            spike_out += x_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            output_record[i, :] = torch.transpose(pred, 1, 0)

        return spike_out, output_record


class SpikingHighwayFCN1(HighwayFCN1):
    def __init__(self, input_shape, num_classes, spiking_neuron, if_param, device, seq_length, max_firing_rate, dt):
        super(SpikingHighwayFCN1, self).__init__(input_shape=input_shape, num_classes=num_classes)
        self.spiking_linear1 = SpikingFeedForwardCell(spiking_op=self.linear1, spiking_neuron=spiking_neuron(
            shape=(128, ), if_params=if_param))
        self.spiking_linear2 = SpikingFeedForwardCell(spiking_op=self.linear2, spiking_neuron=spiking_neuron(
            shape=(128, ), if_params=if_param))
        self.spiking_linear3 = SpikingFeedForwardCell(spiking_op=self.linear3, spiking_neuron=spiking_neuron(
            shape=(256, ), if_params=if_param))
        self.spiking_linear4 = SpikingFeedForwardCell(spiking_op=self.linear4, spiking_neuron=spiking_neuron(
            shape=(num_classes,), if_params=if_param))

        self.seq_length = seq_length
        self.max_firing_rate = max_firing_rate
        self.dt = dt
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        self.spiking_linear3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear4.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear2.initial_state(batch_size=batch_size, device=self.device)
        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        output_record = torch.zeros((self.seq_length, batch_size), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            inp_image = threshold(x, spike_snapshot)

            x_spike = self.spiking_linear1.forward(inp_image)
            r_spike = self.spiking_linear2.forward(x_spike)
            x_spike += r_spike
            x_spike = threshold(x_spike, 1)
            x_spike = self.spiking_linear3.forward(x_spike)
            x_spike = self.spiking_linear4.forward(x_spike)
            spike_out += x_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            output_record[i, :] = torch.transpose(pred, 1, 0)

        return spike_out, output_record


class SpikingFCN2(FCN2):
    def __init__(self, input_shape, num_classes, spiking_neuron, if_param, device, seq_length, max_firing_rate, dt):
        super(SpikingFCN2, self).__init__(input_shape=input_shape, num_classes=num_classes)
        self.spiking_linear1 = SpikingFeedForwardCell(spiking_op=self.linear1, spiking_neuron=spiking_neuron(
            shape=(128, ), if_params=if_param))
        self.spiking_linear2_0 = SpikingFeedForwardCell(spiking_op=self.linear2_0, spiking_neuron=spiking_neuron(
            shape=(128, ), if_params=if_param))
        self.spiking_linear2_1 = SpikingFeedForwardCell(spiking_op=self.linear2_1, spiking_neuron=spiking_neuron(
            shape=(128,), if_params=if_param))
        self.spiking_linear3 = SpikingFeedForwardCell(spiking_op=self.linear3, spiking_neuron=spiking_neuron(
            shape=(256, ), if_params=if_param))
        self.spiking_linear4 = SpikingFeedForwardCell(spiking_op=self.linear4, spiking_neuron=spiking_neuron(
            shape=(num_classes,), if_params=if_param))

        self.seq_length = seq_length
        self.max_firing_rate = max_firing_rate
        self.dt = dt
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        self.spiking_linear3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear4.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear2_0.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear2_1.initial_state(batch_size=batch_size, device=self.device)
        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        output_record = torch.zeros((self.seq_length, batch_size), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            inp_image = threshold(x, spike_snapshot)

            x_spike = self.spiking_linear1.forward(inp_image)
            x_spike = self.spiking_linear2_0.forward(x_spike)
            x_spike = self.spiking_linear2_1.forward(x_spike)
            x_spike = self.spiking_linear3.forward(x_spike)
            x_spike = self.spiking_linear4.forward(x_spike)
            spike_out += x_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            output_record[i, :] = torch.transpose(pred, 1, 0)

        return spike_out, output_record


class SpikingHighwayFCN2(HighwayFCN2):
    def __init__(self, input_shape, num_classes, spiking_neuron, if_param, device, seq_length, max_firing_rate, dt):
        super(SpikingHighwayFCN2, self).__init__(input_shape=input_shape, num_classes=num_classes)
        self.spiking_linear1 = SpikingFeedForwardCell(spiking_op=self.linear1, spiking_neuron=spiking_neuron(
            shape=(128, ), if_params=if_param))
        self.spiking_linear2_0 = SpikingFeedForwardCell(spiking_op=self.linear2_0, spiking_neuron=spiking_neuron(
            shape=(128, ), if_params=if_param))
        self.spiking_linear2_1 = SpikingFeedForwardCell(spiking_op=self.linear2_1, spiking_neuron=spiking_neuron(
            shape=(128,), if_params=if_param))
        self.spiking_linear3 = SpikingFeedForwardCell(spiking_op=self.linear3, spiking_neuron=spiking_neuron(
            shape=(256, ), if_params=if_param))
        self.spiking_linear4 = SpikingFeedForwardCell(spiking_op=self.linear4, spiking_neuron=spiking_neuron(
            shape=(num_classes,), if_params=if_param))

        self.seq_length = seq_length
        self.max_firing_rate = max_firing_rate
        self.dt = dt
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        self.spiking_linear3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear4.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear2_0.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear2_1.initial_state(batch_size=batch_size, device=self.device)
        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        output_record = torch.zeros((self.seq_length, batch_size), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            inp_image = threshold(x, spike_snapshot)

            x_spike = self.spiking_linear1.forward(inp_image)
            r_spike = self.spiking_linear2_0.forward(x_spike)
            r_spike = self.spiking_linear2_1.forward(r_spike)
            x_spike += r_spike
            x_spike = threshold(x_spike, 1)
            x_spike = self.spiking_linear3.forward(x_spike)
            x_spike = self.spiking_linear4.forward(x_spike)
            spike_out += x_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            output_record[i, :] = torch.transpose(pred, 1, 0)

        return spike_out, output_record


class SpikingCNN3(CNN3):
    def __init__(self, input_shape, num_classes, spiking_neuron, if_param, device, seq_length, max_firing_rate, dt):
        super(SpikingCNN3, self).__init__(input_shape=input_shape, num_classes=num_classes)
        self.spiking_conv1 = SpikingFeedForwardCell(spiking_op=self.conv1, spiking_neuron=spiking_neuron(
            shape=(self.conv1.out_channels, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv2 = SpikingFeedForwardCell(spiking_op=self.conv2, spiking_neuron=spiking_neuron(
            shape=(self.conv2.out_channels, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv3 = SpikingFeedForwardCell(spiking_op=self.conv3, spiking_neuron=spiking_neuron(
            shape=(self.conv3.out_channels, self.input_shape[0]//2, self.input_shape[1]//2), if_params=if_param))
        self.spiking_conv4_0 = SpikingFeedForwardCell(spiking_op=self.conv4_0, spiking_neuron=spiking_neuron(
            shape=(self.conv4_0.out_channels, self.input_shape[0] // 2, self.input_shape[1] // 2), if_params=if_param))
        self.spiking_conv4_1 = SpikingFeedForwardCell(spiking_op=self.conv4_1, spiking_neuron=spiking_neuron(
            shape=(self.conv4_1.out_channels, self.input_shape[0] // 2, self.input_shape[1] // 2), if_params=if_param))
        self.spiking_conv5 = SpikingFeedForwardCell(spiking_op=self.conv5, spiking_neuron=spiking_neuron(
            shape=(self.conv5.out_channels, self.input_shape[0]//4, self.input_shape[1]//4), if_params=if_param))
        self.spiking_conv6 = SpikingFeedForwardCell(spiking_op=self.conv6, spiking_neuron=spiking_neuron(
            shape=(self.conv6.out_channels, self.input_shape[0]//4, self.input_shape[1]//4), if_params=if_param))
        self.spiking_linear1 = SpikingFeedForwardCell(spiking_op=self.linear1, spiking_neuron=spiking_neuron(
            shape=(self.linear1.out_features,), if_params=if_param))
        self.spiking_linear2 = SpikingFeedForwardCell(spiking_op=self.linear2, spiking_neuron=spiking_neuron(
            shape=(num_classes,), if_params=if_param))

        self.seq_length = seq_length
        self.max_firing_rate = max_firing_rate
        self.dt = dt
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        self.spiking_conv1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4_0.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4_1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv5.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv6.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear2.initial_state(batch_size=batch_size, device=self.device)
        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        output_record = torch.zeros((self.seq_length, batch_size), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            inp_image = threshold(x, spike_snapshot)

            x_spike = self.spiking_conv1.forward(inp_image)
            x_spike = self.spiking_conv2.forward(x_spike)
            x_spike = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x_spike)
            x_spike = self.spiking_conv3.forward(x_spike)
            x_spike = self.spiking_conv4_0.forward(x_spike)
            x_spike = self.spiking_conv4_1.forward(x_spike)
            x_spike = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x_spike)
            x_spike = self.spiking_conv5.forward(x_spike)
            x_spike = self.spiking_conv6.forward(x_spike)
            x_spike = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x_spike)
            x_spike = x_spike.view(-1, self.num_flat_features(x_spike))
            x_spike = self.spiking_linear1.forward(x_spike)
            x_spike = self.spiking_linear2.forward(x_spike)

            spike_out += x_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            output_record[i, :] = torch.transpose(pred, 1, 0)

        return spike_out, output_record


class SpikingHighwayCNN3(HighwayCNN3):
    def __init__(self, input_shape, num_classes, spiking_neuron, if_param, device, seq_length, max_firing_rate, dt):
        super(SpikingHighwayCNN3, self).__init__(input_shape=input_shape, num_classes=num_classes)
        self.spiking_conv1 = SpikingFeedForwardCell(spiking_op=self.conv1, spiking_neuron=spiking_neuron(
            shape=(self.conv1.out_channels, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv2 = SpikingFeedForwardCell(spiking_op=self.conv2, spiking_neuron=spiking_neuron(
            shape=(self.conv2.out_channels, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv3 = SpikingFeedForwardCell(spiking_op=self.conv3, spiking_neuron=spiking_neuron(
            shape=(self.conv3.out_channels, self.input_shape[0]//2, self.input_shape[1]//2), if_params=if_param))
        self.spiking_conv4_0 = SpikingFeedForwardCell(spiking_op=self.conv4_0, spiking_neuron=spiking_neuron(
            shape=(self.conv4_0.out_channels, self.input_shape[0] // 2, self.input_shape[1] // 2), if_params=if_param))
        self.spiking_conv4_1 = SpikingFeedForwardCell(spiking_op=self.conv4_1, spiking_neuron=spiking_neuron(
            shape=(self.conv4_1.out_channels, self.input_shape[0] // 2, self.input_shape[1] // 2), if_params=if_param))
        self.spiking_conv5 = SpikingFeedForwardCell(spiking_op=self.conv5, spiking_neuron=spiking_neuron(
            shape=(self.conv5.out_channels, self.input_shape[0]//4, self.input_shape[1]//4), if_params=if_param))
        self.spiking_conv6 = SpikingFeedForwardCell(spiking_op=self.conv6, spiking_neuron=spiking_neuron(
            shape=(self.conv6.out_channels, self.input_shape[0]//4, self.input_shape[1]//4), if_params=if_param))
        self.spiking_linear1 = SpikingFeedForwardCell(spiking_op=self.linear1, spiking_neuron=spiking_neuron(
            shape=(self.linear1.out_features,), if_params=if_param))
        self.spiking_linear2 = SpikingFeedForwardCell(spiking_op=self.linear2, spiking_neuron=spiking_neuron(
            shape=(num_classes,), if_params=if_param))

        self.seq_length = seq_length
        self.max_firing_rate = max_firing_rate
        self.dt = dt
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        self.spiking_conv1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4_0.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4_1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv5.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv6.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear2.initial_state(batch_size=batch_size, device=self.device)
        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        output_record = torch.zeros((self.seq_length, batch_size), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            inp_image = threshold(x, spike_snapshot)

            x_spike = self.spiking_conv1.forward(inp_image)
            r_spike = self.spiking_conv2.forward(x_spike)
            x_spike = x_spike + r_spike
            x_spike = threshold(x_spike, 1)
            x_spike = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x_spike)
            x_spike = self.spiking_conv3.forward(x_spike)
            r_spike = self.spiking_conv4_0.forward(x_spike)
            r_spike = self.spiking_conv4_1.forward(r_spike)
            x_spike = r_spike + x_spike
            x_spike = threshold(x_spike, 1)
            x_spike = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x_spike)
            x_spike = self.spiking_conv5.forward(x_spike)
            r_spike = self.spiking_conv6.forward(x_spike)
            x_spike = r_spike + x_spike
            x_spike = threshold(x_spike, 1)
            x_spike = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x_spike)
            x_spike = x_spike.view(-1, self.num_flat_features(x_spike))
            x_spike = self.spiking_linear1.forward(x_spike)
            x_spike = self.spiking_linear2.forward(x_spike)

            spike_out += x_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            output_record[i, :] = torch.transpose(pred, 1, 0)

        return spike_out, output_record


class SpikingCNN_18(CNN_18):
    def __init__(self, input_shape, num_classes, spiking_neuron, if_param, device, seq_length, max_firing_rate, dt):
        super(SpikingCNN_18, self).__init__(input_shape=input_shape, num_classes=num_classes)
        self.spiking_conv1 = SpikingFeedForwardCell(spiking_op=self.conv1, spiking_neuron=spiking_neuron(
            shape=(self.conv1.out_channels, self.input_shape[0], self.input_shape[1]), if_params=if_param))

        self.spiking_conv1_1 = SpikingFeedForwardCell(spiking_op=self.conv1_1, spiking_neuron=spiking_neuron(
            shape=(self.conv1_1.out_channels, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv1_2 = SpikingFeedForwardCell(spiking_op=self.conv1_2, spiking_neuron=spiking_neuron(
            shape=(self.conv1_2.out_channels, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv1_3 = SpikingFeedForwardCell(spiking_op=self.conv1_3, spiking_neuron=spiking_neuron(
            shape=(self.conv1_3.out_channels, self.input_shape[0], self.input_shape[1]), if_params=if_param))

        self.spiking_conv2_1 = SpikingFeedForwardCell(spiking_op=self.conv2_1, spiking_neuron=spiking_neuron(
            shape=(self.conv2_1.out_channels, self.input_shape[0] // 2, self.input_shape[1] // 2), if_params=if_param))
        self.spiking_conv2_2 = SpikingFeedForwardCell(spiking_op=self.conv2_2, spiking_neuron=spiking_neuron(
            shape=(self.conv2_2.out_channels, self.input_shape[0] // 2, self.input_shape[1] // 2), if_params=if_param))
        self.spiking_conv2_3 = SpikingFeedForwardCell(spiking_op=self.conv2_3, spiking_neuron=spiking_neuron(
            shape=(self.conv2_3.out_channels, self.input_shape[0] // 2, self.input_shape[1] // 2), if_params=if_param))
        self.spiking_conv2_4 = SpikingFeedForwardCell(spiking_op=self.conv2_4, spiking_neuron=spiking_neuron(
            shape=(self.conv2_4.out_channels, self.input_shape[0] // 2, self.input_shape[1] // 2), if_params=if_param))
        self.spiking_conv2_5 = SpikingFeedForwardCell(spiking_op=self.conv2_5, spiking_neuron=spiking_neuron(
            shape=(self.conv2_5.out_channels, self.input_shape[0] // 2, self.input_shape[1] // 2),
            if_params=if_param))

        self.spiking_conv3_1 = SpikingFeedForwardCell(spiking_op=self.conv3_1, spiking_neuron=spiking_neuron(
            shape=(self.conv3_1.out_channels, self.input_shape[0] // 4, self.input_shape[1] // 4), if_params=if_param))
        self.spiking_conv3_2 = SpikingFeedForwardCell(spiking_op=self.conv3_2, spiking_neuron=spiking_neuron(
            shape=(self.conv3_2.out_channels, self.input_shape[0] // 4, self.input_shape[1] // 4),
            if_params=if_param))
        self.spiking_conv3_3 = SpikingFeedForwardCell(spiking_op=self.conv3_3, spiking_neuron=spiking_neuron(
            shape=(self.conv3_3.out_channels, self.input_shape[0] // 4, self.input_shape[1] // 4),
            if_params=if_param))
        self.spiking_conv3_4 = SpikingFeedForwardCell(spiking_op=self.conv3_4, spiking_neuron=spiking_neuron(
            shape=(self.conv3_4.out_channels, self.input_shape[0] // 4, self.input_shape[1] // 4), if_params=if_param))
        self.spiking_conv3_5 = SpikingFeedForwardCell(spiking_op=self.conv3_5, spiking_neuron=spiking_neuron(
            shape=(self.conv3_5.out_channels, self.input_shape[0] // 4, self.input_shape[1] // 4),
            if_params=if_param))

        self.spiking_conv4_1 = SpikingFeedForwardCell(spiking_op=self.conv4_1, spiking_neuron=spiking_neuron(
            shape=(self.conv4_1.out_channels, self.input_shape[0] // 8, self.input_shape[1] // 8), if_params=if_param))
        self.spiking_conv4_2 = SpikingFeedForwardCell(spiking_op=self.conv4_2, spiking_neuron=spiking_neuron(
            shape=(self.conv4_2.out_channels, self.input_shape[0] // 8, self.input_shape[1] // 8),
            if_params=if_param))
        self.spiking_conv4_3 = SpikingFeedForwardCell(spiking_op=self.conv4_3, spiking_neuron=spiking_neuron(
            shape=(self.conv4_3.out_channels, self.input_shape[0] // 8, self.input_shape[1] // 8),
            if_params=if_param))
        self.spiking_conv4_4 = SpikingFeedForwardCell(spiking_op=self.conv4_4, spiking_neuron=spiking_neuron(
            shape=(self.conv4_4.out_channels, self.input_shape[0] // 8, self.input_shape[1] // 8), if_params=if_param))
        self.spiking_conv4_5 = SpikingFeedForwardCell(spiking_op=self.conv4_5, spiking_neuron=spiking_neuron(
            shape=(self.conv4_5.out_channels, self.input_shape[0] // 8, self.input_shape[1] // 8),
            if_params=if_param))

        self.spiking_fc = SpikingFeedForwardCell(spiking_op=self.fc, spiking_neuron=spiking_neuron(
            shape=(self.fc.out_features,), if_params=if_param))

        self.seq_length = seq_length
        self.max_firing_rate = max_firing_rate
        self.dt = dt
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        self.spiking_conv1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv1_1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv1_2.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv1_3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2_1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2_2.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2_3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2_4.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2_5.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3_1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3_2.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3_3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3_4.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3_5.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4_1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4_2.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4_3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4_4.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4_5.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_fc.initial_state(batch_size=batch_size, device=self.device)

        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        output_record = torch.zeros((self.seq_length, batch_size), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            inp_image = threshold(x, spike_snapshot)

            out_spike = self.spiking_conv1.forward(inp_image)

            out_spike = self.spiking_conv1_1.forward(out_spike)
            out_spike = self.spiking_conv1_2.forward(out_spike)
            out_spike = self.spiking_conv1_3.forward(out_spike)

            out_spike = self.spiking_conv2_1.forward(out_spike)
            out_spike = self.spiking_conv2_2.forward(out_spike)
            out_spike = self.spiking_conv2_3.forward(out_spike)
            out_spike = self.spiking_conv2_4.forward(out_spike)
            out_spike = self.spiking_conv2_5.forward(out_spike)

            out_spike = self.spiking_conv3_1.forward(out_spike)
            out_spike = self.spiking_conv3_2.forward(out_spike)
            out_spike = self.spiking_conv3_3.forward(out_spike)
            out_spike = self.spiking_conv3_4.forward(out_spike)
            out_spike = self.spiking_conv3_5.forward(out_spike)

            out_spike = self.spiking_conv4_1.forward(out_spike)
            out_spike = self.spiking_conv4_2.forward(out_spike)
            out_spike = self.spiking_conv4_3.forward(out_spike)
            out_spike = self.spiking_conv4_4.forward(out_spike)
            out_spike = self.spiking_conv4_5.forward(out_spike)

            out_spike = torch.nn.functional.avg_pool2d(out_spike, 4)

            out_spike = out_spike.view(-1, self.num_flat_features(out_spike))
            out_spike = self.spiking_fc.forward(out_spike)

            spike_out += out_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            output_record[i, :] = torch.transpose(pred, 1, 0)

        return spike_out, output_record


class SpikingHighwayCNN_18(HighwayCNN_18):
    def __init__(self, input_shape, num_classes, spiking_neuron, if_param, device, seq_length, max_firing_rate, dt):
        super(SpikingHighwayCNN_18, self).__init__(input_shape=input_shape, num_classes=num_classes)
        self.spiking_conv1 = SpikingFeedForwardCell(spiking_op=self.conv1, spiking_neuron=spiking_neuron(
            shape=(self.conv1.out_channels, self.input_shape[0], self.input_shape[1]), if_params=if_param))

        self.spiking_conv1_1 = SpikingFeedForwardCell(spiking_op=self.conv1_1, spiking_neuron=spiking_neuron(
            shape=(self.conv1_1.out_channels, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv1_2 = SpikingFeedForwardCell(spiking_op=self.conv1_2, spiking_neuron=spiking_neuron(
            shape=(self.conv1_2.out_channels, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv1_3 = SpikingFeedForwardCell(spiking_op=self.conv1_3, spiking_neuron=spiking_neuron(
            shape=(self.conv1_3.out_channels, self.input_shape[0], self.input_shape[1]), if_params=if_param))

        self.spiking_conv2_1 = SpikingFeedForwardCell(spiking_op=self.conv2_1, spiking_neuron=spiking_neuron(
            shape=(self.conv2_1.out_channels, self.input_shape[0] // 2, self.input_shape[1] // 2), if_params=if_param))
        self.spiking_conv2_2 = SpikingFeedForwardCell(spiking_op=self.conv2_2, spiking_neuron=spiking_neuron(
            shape=(self.conv2_2.out_channels, self.input_shape[0] // 2, self.input_shape[1] // 2), if_params=if_param))
        self.spiking_conv2_3 = SpikingFeedForwardCell(spiking_op=self.conv2_3, spiking_neuron=spiking_neuron(
            shape=(self.conv2_3.out_channels, self.input_shape[0] // 2, self.input_shape[1] // 2), if_params=if_param))
        self.spiking_conv2_4 = SpikingFeedForwardCell(spiking_op=self.conv2_4, spiking_neuron=spiking_neuron(
            shape=(self.conv2_4.out_channels, self.input_shape[0] // 2, self.input_shape[1] // 2), if_params=if_param))
        self.spiking_conv2_5 = SpikingFeedForwardCell(spiking_op=self.conv2_5, spiking_neuron=spiking_neuron(
            shape=(self.conv2_5.out_channels, self.input_shape[0] // 2, self.input_shape[1] // 2),
            if_params=if_param))

        self.spiking_conv3_1 = SpikingFeedForwardCell(spiking_op=self.conv3_1, spiking_neuron=spiking_neuron(
            shape=(self.conv3_1.out_channels, self.input_shape[0] // 4, self.input_shape[1] // 4), if_params=if_param))
        self.spiking_conv3_2 = SpikingFeedForwardCell(spiking_op=self.conv3_2, spiking_neuron=spiking_neuron(
            shape=(self.conv3_2.out_channels, self.input_shape[0] // 4, self.input_shape[1] // 4),
            if_params=if_param))
        self.spiking_conv3_3 = SpikingFeedForwardCell(spiking_op=self.conv3_3, spiking_neuron=spiking_neuron(
            shape=(self.conv3_3.out_channels, self.input_shape[0] // 4, self.input_shape[1] // 4),
            if_params=if_param))
        self.spiking_conv3_4 = SpikingFeedForwardCell(spiking_op=self.conv3_4, spiking_neuron=spiking_neuron(
            shape=(self.conv3_4.out_channels, self.input_shape[0] // 4, self.input_shape[1] // 4), if_params=if_param))
        self.spiking_conv3_5 = SpikingFeedForwardCell(spiking_op=self.conv3_5, spiking_neuron=spiking_neuron(
            shape=(self.conv3_5.out_channels, self.input_shape[0] // 4, self.input_shape[1] // 4),
            if_params=if_param))

        self.spiking_conv4_1 = SpikingFeedForwardCell(spiking_op=self.conv4_1, spiking_neuron=spiking_neuron(
            shape=(self.conv4_1.out_channels, self.input_shape[0] // 8, self.input_shape[1] // 8), if_params=if_param))
        self.spiking_conv4_2 = SpikingFeedForwardCell(spiking_op=self.conv4_2, spiking_neuron=spiking_neuron(
            shape=(self.conv4_2.out_channels, self.input_shape[0] // 8, self.input_shape[1] // 8),
            if_params=if_param))
        self.spiking_conv4_3 = SpikingFeedForwardCell(spiking_op=self.conv4_3, spiking_neuron=spiking_neuron(
            shape=(self.conv4_3.out_channels, self.input_shape[0] // 8, self.input_shape[1] // 8),
            if_params=if_param))
        self.spiking_conv4_4 = SpikingFeedForwardCell(spiking_op=self.conv4_4, spiking_neuron=spiking_neuron(
            shape=(self.conv4_4.out_channels, self.input_shape[0] // 8, self.input_shape[1] // 8), if_params=if_param))
        self.spiking_conv4_5 = SpikingFeedForwardCell(spiking_op=self.conv4_5, spiking_neuron=spiking_neuron(
            shape=(self.conv4_5.out_channels, self.input_shape[0] // 8, self.input_shape[1] // 8),
            if_params=if_param))

        self.spiking_fc = SpikingFeedForwardCell(spiking_op=self.fc, spiking_neuron=spiking_neuron(
            shape=(self.fc.out_features,), if_params=if_param))

        self.seq_length = seq_length
        self.max_firing_rate = max_firing_rate
        self.dt = dt
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        self.spiking_conv1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv1_1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv1_2.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv1_3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2_1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2_2.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2_3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2_4.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2_5.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3_1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3_2.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3_3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3_4.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv3_5.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4_1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4_2.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4_3.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4_4.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv4_5.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_fc.initial_state(batch_size=batch_size, device=self.device)

        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        output_record = torch.zeros((self.seq_length, batch_size), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            inp_image = threshold(x, spike_snapshot)

            x_spike = self.spiking_conv1.forward(inp_image)

            x_spike = self.spiking_conv1_1.forward(x_spike)
            x_spike = self.spiking_conv1_2.forward(x_spike)
            out_spike = self.spiking_conv1_3.forward(x_spike)
            x_spike = out_spike + x_spike

            x_spike = self.spiking_conv2_1.forward(x_spike)
            x_spike = self.spiking_conv2_2.forward(x_spike)
            out_spike = self.spiking_conv2_3.forward(x_spike)
            x_spike = out_spike + x_spike
            x_spike = self.spiking_conv2_4.forward(x_spike)
            x_spike = self.spiking_conv2_5.forward(x_spike)

            x_spike = self.spiking_conv3_1.forward(x_spike)
            x_spike = self.spiking_conv3_2.forward(x_spike)
            out_spike = self.spiking_conv3_3.forward(x_spike)
            x_spike = out_spike + x_spike
            x_spike = self.spiking_conv3_4.forward(x_spike)
            x_spike = self.spiking_conv3_5.forward(x_spike)

            x_spike = self.spiking_conv4_1.forward(x_spike)
            x_spike = self.spiking_conv4_2.forward(x_spike)
            out_spike = self.spiking_conv4_3.forward(x_spike)
            x_spike = out_spike + x_spike
            x_spike = self.spiking_conv4_4.forward(x_spike)
            x_spike = self.spiking_conv4_5.forward(x_spike)

            x_spike = torch.nn.functional.avg_pool2d(x_spike, 4)

            x_spike = x_spike.view(-1, self.num_flat_features(x_spike))
            x_spike = self.spiking_fc.forward(x_spike)

            spike_out += x_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            output_record[i, :] = torch.transpose(pred, 1, 0)

        return spike_out, output_record

