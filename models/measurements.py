import math
import torch as t 
import torch.nn as nn
import numpy as np 
import random
from .basic_module import BasicModule 
from config import opt
import torchsnooper

class Measurements(BasicModule):
    def __init__(self,k_cr,nonk_cr,Height,Width,seqLength,bernoulli_p=50):
        super(Measurements,self).__init__()

        self.key_cr = k_cr
        self.nonkey_cr = nonk_cr
        self.h = Height
        self.w = Width
        self.s_l = seqLength
        self.b_p = bernoulli_p
        self.in_size = int(self.h*self.w)
        self.key_out_size = int(self.h*self.w*self.key_cr)
        self.nonkey_out_size = int(self.h*self.w*self.nonkey_cr)
        self.key_weight = nn.Parameter(t.Tensor(self.key_out_size,self.in_size)).to(opt.device)
        self.nonkey_weight = nn.Parameter(t.Tensor(self.nonkey_out_size,self.in_size)).to(opt.device)

        self.save_key_weight = self.key_weight.data.clone()
        self.save_nonkey_weight = self.nonkey_weight.data.clone()
        
        self.reset_parameter(bernoulli_p)
        self.binarization()


    def binarization(self):
        self.key_weight.data.clamp(-1.0,1.0)
        self.key_weight.data = 0.5*(self.key_weight.data.sign()+1)
        self.key_weight.data[self.key_weight.data == 0.5] = 1

        self.nonkey_weight.data.clamp(-1.0,1.0)
        self.nonkey_weight.data = 0.5*(self.nonkey_weight.data.sign()+1)
        self.nonkey_weight.data[self.nonkey_weight.data == 0.5] = 1

        self.save_key_weight.copy_(self.key_weight.data)
        self.save_nonkey_weight.copy_(self.nonkey_weight.data)
    
    def reset_parameter(self,p):
        key_bernoulli_weights = t.FloatTensor(
            self.key_weight.size()).bernoulli_(p / 100)
        n = self.s_l
        stdv = 1. / math.sqrt(n)

        weights_zero = key_bernoulli_weights[key_bernoulli_weights ==
                                         0].uniform_(-stdv, 0)
        weights_one = key_bernoulli_weights[key_bernoulli_weights == 1].uniform_(
            0, stdv)
        key_bernoulli_weights[key_bernoulli_weights == 0] = weights_zero
        key_bernoulli_weights[key_bernoulli_weights == 1] = weights_one
        self.key_weight.data.copy_(key_bernoulli_weights)

        nonkey_bernoulli_weights = t.FloatTensor(
            self.nonkey_weight.size()).bernoulli_(p / 100)

        weights_zero = nonkey_bernoulli_weights[nonkey_bernoulli_weights ==
                                         0].uniform_(-stdv, 0)
        weights_one = nonkey_bernoulli_weights[nonkey_bernoulli_weights == 1].uniform_(
            0, stdv)
        nonkey_bernoulli_weights[nonkey_bernoulli_weights == 0] = weights_zero
        nonkey_bernoulli_weights[nonkey_bernoulli_weights == 1] = weights_one
        self.nonkey_weight.data.copy_(nonkey_bernoulli_weights)
    
    def restore(self):
        self.key_weight.data.copy_(self.save_key_weight)
        self.nonkey_weight.data.copy_(self.save_nonkey_weight)
    #@torchsnooper.snoop()
    def forward(self,input):
        #input data type torch tensor
        #input data size [batch_size,seqLength,height,width]
        #output type tuple(torch tensor)
        #output size ([batch_size,key_out_size],[batch_size,seqLength-1,nonkey_out_size])
        b_s = input.shape[0]
        input_ = input.view(b_s,self.s_l,self.in_size).unsqueeze(3)
        key_out = t.zeros(b_s,self.key_out_size).to(opt.device)
        nonkey_out = t.zeros(b_s,self.s_l-1,self.nonkey_out_size).to(opt.device)
        key_weight_ = self.key_weight.repeat(b_s,1,1)
        nonkey_weight_ = self.nonkey_weight.repeat(b_s,1,1) 

        key_out = t.bmm(key_weight_,input_[:,0,:]).view(b_s,self.key_out_size)
        for i in range(self.s_l-1):
            nonkey_out[:,i,:] = t.bmm(nonkey_weight_,input_[:,i+1,:]).view(b_s,self.nonkey_out_size)
        return key_out,nonkey_out
'''
CR = [1,0.5]
m = Measurements(20,CR,32,32,10,50)
input = np.random.rand(20,10,32,32)
input_ = t.Tensor(input)
b,c = m.forward(input_)
print(b.size())
print(c.size())
'''
