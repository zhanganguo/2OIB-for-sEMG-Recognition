from typing import NamedTuple
from collections import Iterable

import torch

from snn.utils import threshold


class AbstractSpikingNeuron:
    def __init__(self):
        pass

    def initialize_state(self, *kwargs):
        pass

    def forward(self, *kwargs):
        pass


class IFParameters(NamedTuple):
    """Parametrization of a LIF neuron

    Parameters:
        tau_mem (torch.Tensor): membrane time constant)
        resistance_mem (torch.Tensor): synaptic time constant)
        v_th (torch.Tensor): threshhold potential
        v_reset (torch.Tensor): reset potential
    """
    tau_mem: torch.Tensor = torch.as_tensor(1.0)
    resistance_mem: torch.Tensor = torch.as_tensor(1.0)
    capacity_mem: torch.Tensor = torch.as_tensor(1.0)
    v_th: torch.Tensor = torch.as_tensor(3.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    t_refrac: torch.Tensor = torch.as_tensor(0.0)


class IfNeuron(AbstractSpikingNeuron):
    def __init__(self, shape, if_params: IFParameters):
        super(AbstractSpikingNeuron, self).__init__()
        self.shape = shape
        self.if_parameters = if_params

        self.v_mem = None

    def initialize_state(self, batch_size, device):
        self.v_mem = torch.zeros(batch_size, *self.shape, device=device)

    def forward(self, input_current):
        self.v_mem += input_current

        spike = threshold(self.v_mem, self.if_parameters.v_th)
        self.v_mem = (1 - spike) * self.v_mem + spike * self.if_parameters.v_reset
        return spike


class IfNeuron_with_IP(AbstractSpikingNeuron):
    def __init__(self, shape, if_params: IFParameters):
        super(AbstractSpikingNeuron, self).__init__()
        self.shape = shape
        self.if_parameters = if_params

        self.beta = 0.6
        self.sigma = 0.5
        self.tau = 0.01

        self.v_mem = None
        self.rC = None
        # self.rR = None

    def initialize_state(self, batch_size, device):
        self.v_mem = torch.zeros(batch_size, *self.shape, device=device)
        self.rC = torch.ones(batch_size, *self.shape, device=device) * 2
        # self.rR = torch.ones(batch_size, *self.shape, device=device) * 2

    def forward(self, input_current):
        self.v_mem += input_current * self.rC

        spike = threshold(self.v_mem, self.if_parameters.v_th)
        delta_rC = 1.0 / self.rC - self.sigma * spike * input_current + self.beta * (1 - self.sigma * spike) * input_current
        # delta_rR = self.tau * (-self.rR + self.sigma * spike - self.beta * (1 - self.sigma * spike))
        self.rC += self.tau * delta_rC
        # self.rR += delta_rR
        return spike


class IfNeuron_with_SelfDriven_IP(IfNeuron_with_IP):
    def __init__(self, shape, if_params: IFParameters):
        super(IfNeuron_with_IP, self).__init__()

    def forward(self, input_current):
        self.v_mem += input_current * self.rC

        spike = threshold(self.v_mem, self.if_parameters.v_th)

        delta_rC = 1.0 / self.rC - self.sigma * spike * input_current + self.beta * (1 - self.sigma * spike) * input_current
        # delta_rR = self.tau * (-self.rR + self.sigma * spike - self.beta * (1 - self.sigma * spike))
        self.rC += self.tau * delta_rC * spike
        # self.rR += delta_rR
        return spike


class IfNeuron_with_InputDriven_IP(IfNeuron_with_IP):
    def __init__(self, shape, if_params: IFParameters):
        super(IfNeuron_with_IP, self).__init__()

    def forward(self, input_current):
        self.v_mem += input_current * self.rC

        spike = threshold(self.v_mem, self.if_parameters.v_th)

        input_event = torch.where(
            input_current > 0.001,      # Use a small positive value instead of zero.
            torch.zeros_like(input_current),
            torch.ones_like(input_current),
        )

        delta_rC = 1.0 / self.rC - self.sigma * spike * input_current + self.beta * (1 - self.sigma * spike) * input_current
        # delta_rR = self.tau * (-self.rR + self.sigma * spike - self.beta * (1 - self.sigma * spike))
        self.rC += self.tau * delta_rC * input_event
        # self.rR += delta_rR
        return spike

