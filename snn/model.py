import torch
from ann.model import FCN
from snn.spiking_neuron import IfNeuron
from snn.spiking_operation import SpikingFeedForwardCell
from snn.utils import threshold


class SpikingFCN(FCN):
    def __init__(self, input_shape, num_classes, spiking_neuron, if_params, device, seq_length, max_firing_rate, dt):
        super(SpikingFCN, self).__init__(input_shape=input_shape, num_classes=num_classes)
        self.spiking_conv1 = SpikingFeedForwardCell(spiking_op=self.conv1, spiking_neuron=spiking_neuron(
            shape=(self.conv1.out_channels, input_shape[1], input_shape[2]), if_params=if_params))
        self.spiking_conv2 = SpikingFeedForwardCell(spiking_op=self.conv2, spiking_neuron=spiking_neuron(
            shape=(self.conv2.out_channels, input_shape[1], input_shape[2]), if_params=if_params))
        self.spiking_conv3 = SpikingFeedForwardCell(spiking_op=self.conv3, spiking_neuron=spiking_neuron(
            shape=(self.conv3.out_channels, input_shape[1]//2, input_shape[2]//2), if_params=if_params))
        self.spiking_conv4 = SpikingFeedForwardCell(spiking_op=self.conv4, spiking_neuron=spiking_neuron(
            shape=(self.conv4.out_channels, input_shape[1]//2, input_shape[2]//2), if_params=if_params))
        self.spiking_conv5 = SpikingFeedForwardCell(spiking_op=self.conv5, spiking_neuron=spiking_neuron(
            shape=(self.conv5.out_channels, 3, 3), if_params=if_params))
        self.spiking_conv6 = SpikingFeedForwardCell(spiking_op=self.conv6, spiking_neuron=spiking_neuron(
            shape=(self.conv6.out_channels, 3, 3), if_params=if_params))

        self.spiking_linear_0 = SpikingFeedForwardCell(spiking_op=self.linear_0, spiking_neuron=spiking_neuron(shape=(
            self.conv6.out_channels*2,), if_params=if_params))
        self.spiking_linear = SpikingFeedForwardCell(spiking_op=self.linear, spiking_neuron=spiking_neuron(shape=(
            num_classes,), if_params=if_params))

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
        self.spiking_conv5.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv6.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear_0.initial_state(batch_size=batch_size, device=self.device)
        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        output_record = torch.zeros((self.seq_length, batch_size), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            inp_image = threshold(x, spike_snapshot)

            x_spike = self.spiking_conv1.forward(inp_image)
            x_spike = self.spiking_conv2.forward(x_spike)
            x_spike = self.spiking_conv3.forward(x_spike)
            x_spike = self.spiking_conv4.forward(x_spike)
            x_spike = self.spiking_conv5.forward(x_spike)
            x_spike = self.spiking_conv6.forward(x_spike)
            x_spike = x_spike.view(-1, self.num_flat_features(x_spike))
            x_spike = self.spiking_linear_0.forward(x_spike)
            x_spike = self.spiking_linear.forward(x_spike)

            spike_out += x_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            output_record[i, :] = torch.transpose(pred, 1, 0)

        return spike_out, output_record

