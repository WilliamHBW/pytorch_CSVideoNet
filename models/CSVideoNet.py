import torch as t 
import torch.nn as nn
from collections import OrderedDict
from .basic_module import BasicModule 
from .measurements import Measurements
import time
import torchsnooper
from config import opt
import torchsnooper

class KeyCNN(BasicModule):
    def __init__(self,key_CR,Height,Width,numChannels):
        super(KeyCNN,self).__init__()

        self.key_CR = key_CR
        self.Height = Height
        self.Width = Width
        self.numChannels = numChannels

        self.fc1 = nn.Linear(int(self.key_CR*self.Height*self.Width),self.Height*self.Width)
        cnn_layers = OrderedDict()
        cnn_layers['conv1'] = nn.Conv2d(1,128,1,1,0)
        cnn_layers['relu1'] = nn.ReLU(inplace=True)
        cnn_layers['conv2'] = nn.Conv2d(128,64,1,1,0)
        cnn_layers['relu2'] = nn.ReLU(inplace=True)
        cnn_layers['conv3'] = nn.Conv2d(64,32,3,1,1)
        cnn_layers['relu3'] = nn.ReLU(inplace=True)
        cnn_layers['conv4'] = nn.Conv2d(32,32,3,1,1)
        cnn_layers['relu4'] = nn.ReLU(inplace=True)
        cnn_layers['conv5'] = nn.Conv2d(32,16,3,1,1)
        cnn_layers['relu5'] = nn.ReLU(inplace=True)
        cnn_layers['conv6'] = nn.Conv2d(16,16,3,1,1)
        cnn_layers['relu6'] = nn.ReLU(inplace=True)
        cnn_layers['conv7'] = nn.Conv2d(16,1,3,1,1)
        self.key_cnn = nn.Sequential(cnn_layers)
    #@torchsnooper.snoop()
    def forward(self,input):
        #input size [batch_size,key_CR*Height*Width]
        #output size [batch_size,height*width]
        b_s = input.size(0)
        x_1 = self.fc1(input).view(b_s,self.Height,self.Width).unsqueeze(1)
        output = self.key_cnn(x_1).view(b_s,self.Height*self.Width)
        return output

class nonKeyCNN(BasicModule):
    def __init__(self,nonkey_CR,Height,Width,numChannels):
        super(nonKeyCNN,self).__init__()

        self.nonkey_CR = nonkey_CR
        self.Height = Height
        self.Width = Width
        self.numChannels = numChannels
        self.numHidden = {}
        self.numHidden[0] = 64
        self.numHidden[1] = 32
        self.numHidden[2] = 1

        self.fc1 = nn.Linear(int(self.nonkey_CR*self.Height*self.Width),self.Height*self.Width)
        cnn_layers = OrderedDict()
        cnn_layers['conv1'] = nn.Conv2d(self.numChannels,self.numHidden[0],3,1,1)
        cnn_layers['relu1'] = nn.ReLU(inplace=True)
        cnn_layers['conv2'] = nn.Conv2d(self.numHidden[0],self.numHidden[1],3,1,1)
        cnn_layers['relu2'] = nn.ReLU(inplace=True)
        self.nonkey_cnn = nn.Sequential(cnn_layers)
        self.fc2 = nn.Linear(self.numHidden[1]*self.Height*self.Width,self.numHidden[2]*self.Height*self.Width)
    #@torchsnooper.snoop()
    def forward(self,input):
        #input size [batch_size,nonkey_CR*Height*Width]
        #output size [batch_size,height*width]
        b_s = input.size(0)
        x_1 = self.fc1(input).view(b_s,self.Height,self.Width).unsqueeze(1)
        x_2 = self.nonkey_cnn(x_1).view(b_s,self.numHidden[1]*self.Height*self.Width)
        output = self.fc2(x_2).view(b_s,self.Height*self.Width)
        return output
    
class RNN_(BasicModule):
    def __init__(self,seqLength,Height,Width,numChannels):
        super(RNN_,self).__init__()
        
        self.seqLength = seqLength
        self.Height = Height
        self.Width = Width
        self.input_size = 1*self.Height*self.Width
        self.numHidden = {}
        self.numHidden[0] = 6*self.Height*self.Width
        self.numHidden[1] = 6*self.Height*self.Width
        self.numHidden[2] = numChannels*self.Height*self.Width
        
        self.rnn_1 = nn.GRU(self.input_size,self.numHidden[0],1)
        self.rnn_2 = nn.GRU(self.numHidden[0],self.numHidden[1],1)
        self.rnn_3 = nn.GRU(self.numHidden[1],self.numHidden[2],1)
    
    def forward(self,input):
        #input size [seqLength,batch_size,height*width]
        #output size [batch_size,seqLength,height,width]
        if not hasattr(self, '_flattened'):
            self.rnn_1.flatten_parameters()
            self.rnn_2.flatten_parameters()
            self.rnn_3.flatten_parameters()
        setattr(self, '_flattened', True)
        b_s = input.size(1)
        x_1,_ = self.rnn_1(input)
        x_2,_= self.rnn_2(x_1)
        x_3,_ = self.rnn_3(x_2)

        output = x_3.view(self.seqLength,b_s,self.Height,self.Width)
        output_ = output.permute(1,0,2,3)
        return output_

class CSVideoNet(BasicModule):
    def __init__(self,CR,seqLength,Height,Width,numChannels,bernoulli_p=50):
        super(CSVideoNet,self).__init__()
        self.key_CR = CR[0]
        self.nonkey_CR = CR[1]
        self.seqLength = seqLength
        self.Height = Height
        self.Width = Width
        self.numChannels = numChannels
        self.p = bernoulli_p

        #self.measurement = Measurements(self.key_CR,self.nonkey_CR,self.Height,self.Width,self.seqLength,self.p)
        self.rnnmodule = RNN_(self.seqLength,self.Height,self.Width,self.numChannels)
        self.key_cnnmodule = KeyCNN(self.key_CR,self.Height,self.Width,self.numChannels)
        self.nonkey_cnnmodule_1 = nonKeyCNN(self.nonkey_CR,self.Height,self.Width,self.numChannels)
        #self.nonkey_cnnmodule_2 = nonKeyCNN(self.nonkey_CR,self.Height,self.Width,self.numChannels)
        #self.nonkey_cnnmodule_3 = nonKeyCNN(self.nonkey_CR,self.Height,self.Width,self.numChannels)
        #self.nonkey_cnnmodule_4 = nonKeyCNN(self.nonkey_CR,self.Height,self.Width,self.numChannels)
        #self.nonkey_cnnmodule_5 = nonKeyCNN(self.nonkey_CR,self.Height,self.Width,self.numChannels)
        #self.nonkey_cnnmodule_6 = nonKeyCNN(self.nonkey_CR,self.Height,self.Width,self.numChannels)
        #self.nonkey_cnnmodule_7 = nonKeyCNN(self.nonkey_CR,self.Height,self.Width,self.numChannels)
        #self.nonkey_cnnmodule_8 = nonKeyCNN(self.nonkey_CR,self.Height,self.Width,self.numChannels)
        #self.nonkey_cnnmodule_9 = nonKeyCNN(self.nonkey_CR,self.Height,self.Width,self.numChannels)
    #@torchsnooper.snoop()
    def recon_forward(self,x_key,x_nonkey):
        #x_key size is [batch_size,keyCR*Height*Width]
        #x_nonkey size is [batch_size,seqLength-1,nonkeyCR*Height*Width]
        #output size [batch_size,seqLength,height,width]
        b_s = x_key.size(0)
        x_med = t.Tensor(b_s,self.seqLength,self.Height*self.Width).to(opt.device)
    
        x_med[:,0] = self.key_cnnmodule(x_key)
        x_med[:,1] = self.nonkey_cnnmodule_1(x_nonkey[:,0])
        #x_med[:,2] = self.nonkey_cnnmodule_2(x_nonkey[:,1])
        #x_med[:,3] = self.nonkey_cnnmodule_3(x_nonkey[:,2])
        #x_med[:,4] = self.nonkey_cnnmodule_4(x_nonkey[:,3])
        #x_med[:,5] = self.nonkey_cnnmodule_5(x_nonkey[:,4])
        #x_med[:,6] = self.nonkey_cnnmodule_6(x_nonkey[:,5])
        #x_med[:,7] = self.nonkey_cnnmodule_7(x_nonkey[:,6])
        #x_med[:,8] = self.nonkey_cnnmodule_8(x_nonkey[:,7])
        #x_med[:,9] = self.nonkey_cnnmodule_9(x_nonkey[:,8])

        x_med_ = x_med.permute(1,0,2)
        output = self.rnnmodule(x_med_)

        return output
    
    def load(self,load_path,train=True):
        if train:
            dic_pre_trained = t.load(load_path)
            trained_list = list(dic_pre_trained.keys())
            dic_key_cnn = self.key_cnnmodule.state_dict().copy()
            key_cnn_list = list(dic_key_cnn.keys())
            for i in range(len(trained_list)):
                dic_key_cnn[key_cnn_list[i]] = dic_pre_trained[trained_list[i]]
            self.key_cnnmodule.load_state_dict(dic_key_cnn)
            print("the key cnn net has been loaded successfully!")
        else:
            dic_pre_trained = t.load(load_path)
            trained_list = list(dic_pre_trained.keys())
            dic_csvideonet = self.state_dict().copy()
            csvideonet_list = list(dic_csvideonet.keys())
            for i in range(len(trained_list)):
                dic_csvideonet[csvideonet_list[i]] = dic_pre_trained[trained_list[i]]
            self.load_state_dict(dic_csvideonet)
            print("the whole net has been loaded successfully!")
            
    '''
    def save(self):
        #self.key_cnnmodule.save('key_cnnmodule')
        #self.nonkey_cnnmodule_1.save('nonkey_cnnmodule')
        self.rnnmodule.save('rnnmodule')
        #self.measurement.save('measurements')
        print("the whole net has been saved successfully!")
    '''
    #@torchsnooper.snoop()
    def forward(self,key_m,non_key_m):
        #input type torch tensor
        #input size [batch_size,seqLength,height,width]
        #output type torch tensor
        #output size [batch_size,seqLength,height,width]
        #key_m,non_key_m = self.measurement(input)
        output = self.recon_forward(key_m,non_key_m)
        return output
'''
CR = [1,0.5]
csnet = CSVideoNet(20,CR,10,32,32,1)        
input = t.rand(20,10,32,32)
print(input.size())
output = csnet(input)
print(output.size())
'''


