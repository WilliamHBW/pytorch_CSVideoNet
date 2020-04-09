import torch as t 
import torch.nn as nn 
from collections import OrderedDict
from .basic_module import BasicModule 
from .measurements import Measurements 
from .CSVideoNet import KeyCNN
import time 
from config import opt 
import torchsnooper

class Key_CSVideoNet(BasicModule):
    def __init__(self,key_CR,Height,Width,numChannels):
        super(Key_CSVideoNet,self).__init__()

        self.key_CR = key_CR
        self.Height = Height
        self.Widht = Width
        self.numChannels = numChannels

        self.KeyCNN = KeyCNN(self.key_CR,self.Height,self.Widht,self.numChannels)
        self.Measurements = Measurements(self.key_CR,self.key_CR,self.Height,self.Widht,2)
    #@torchsnooper.snoop()
    def forward(self,input):
        #input type torch tensor
        #input size [batch_size,1,height,width]
        #output type torch tensor
        #output size [batch_size,1,height,width]
        input_ = input.unsqueeze(1).repeat(1,2,1,1)
        key_m,_ = self.Measurements(input_)
        output = self.KeyCNN(key_m).view(input.size(0),self.Height,self.Widht)
        return output


