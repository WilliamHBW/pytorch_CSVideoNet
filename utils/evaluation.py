import numpy as np 
import math
import torch as t
import cv2
import torchsnooper

class Evaluation(object):
    def __init__(self):
        self.psnr_acc = 0
        self.input_num = 0
    
    def reset(self):
        self.psnr_acc = 0
        self.input_num = 0

    def psnr_cal(self,target,ref):
        # target:目标图像  ref:参考图像
        ref_ = ref.cpu()
        target_ = target.cpu()
        ref_data = ref_.numpy()
        target_data = target_.numpy()
        diff = (ref_data - target_data)
        diff_ = diff.flatten('C')
        #rmse = math.sqrt( np.mean(diff_ ** 2.) )
        rmse = np.mean(diff_**2.)
        return 10*math.log10(1.0/rmse)
    #@torchsnooper.snoop()
    def add(self,target,ref):
        #input type torch tensor
        #input size [batch_size,seqLength,height,width]
        #output size psnr_ave
        psnr_ = 0
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                psnr_ = psnr_ + self.psnr_cal(target[i][j],ref[i][j])
                '''
                if(self.psnr_cal(target[i][j],ref[i][j]) < 5):
                    #print(self.psnr_cal(target[i][j],ref[i][j]))
                    target_addr = "./psnr/target"+str(i)+str(j)+".jpg"
                    ref_addr = "./psnr/ref"+str(i)+str(j)+".jpg"
                    print(target[i][j].size(),ref[i][j].size())
                    cv2.imwrite(target_addr,target[i][j].unsqueeze_(2).numpy())
                    cv2.imwrite(ref_addr,ref[i][j].unsqueeze_(2).numpy())
                '''
        psnr_ = psnr_ / (target.size(0)*target.size(1))
        self.psnr_acc = self.psnr_acc + psnr_
        self.input_num = self.input_num + 1
    
    def psnr_value(self):
        return self.psnr_acc/self.input_num
'''
eval = Evaluation()
input = t.rand(20,10,32,32)
ref = t.rand(20,10,32,32)*1e-2
eval.add(input,input+ref)
re = eval.psnr_value()
print(re)
'''
