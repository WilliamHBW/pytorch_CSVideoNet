import torch as t 
import torch.nn as nn
from collections import OrderedDict
from .basic_module import BasicModule

class SimpleNet(BasicModule):
    def __init__(self,input_size,output_size):
        super(SimpleNet,self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        layers = OrderedDict()
        layers['linear' + str(0)] = nn.Linear(self.input_size,self.output_size)

        self.model = nn.Sequential(layers)
    
    def forward(self,input):
        output = self.model(input)
        return output
