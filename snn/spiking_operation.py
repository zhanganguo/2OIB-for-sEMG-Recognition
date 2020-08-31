from collections import Iterable
from snn.spiking_neuron import AbstractSpikingNeuron


class SpikingFeedForwardCell:
    def __init__(self, spiking_op, spiking_neuron: AbstractSpikingNeuron):
        self.spiking_neuron = spiking_neuron
        self.spiking_op = spiking_op

    def initial_state(self, batch_size, device):
        self.spiking_neuron.initialize_state(batch_size=batch_size, device=device)

    def forward(self, x):
        xs = self.spiking_op(x)
        spike = self.spiking_neuron.forward(xs)
        return spike


