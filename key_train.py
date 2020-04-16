from config import opt
import os 
import torch as t 
import models 
import utils 
from data.dataset_csvideonet import UCF101 
from torch.utils .data import DataLoader
from torchnet import meter 
from tqdm import tqdm 
import math
import numpy as np

def train(**kwargs):
    opt._parse(kwargs)

    model = models.Key_CSVideoNet(opt.CR[0],opt.Height,opt.Width,1)
    #if t.cuda.device_count() > 1:
    #    model = t.nn.DataParallel(model,device_ids = [0,1,2,3])
    model.to(opt.device)

    train_data = UCF101(opt.train_data_root,train=True)
    val_data = UCF101(opt.train_data_root,train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    
    criterion = t.nn.MSELoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),
                           lr = lr,
                           weight_decay = opt.weight_decay)
    
    utils_eval = utils.Evaluation()
    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e10

    for epoch in range(opt.max_epoch):
        loss_meter.reset()

        for ii,data in tqdm(enumerate(train_dataloader)):
            input = data[0].float().to(opt.device).squeeze(1)
            target = input
            input_ = input.view(input.size(0),input.size(1)*input.size(2),1)
            weight = key_bernoulli_weights.repeat(input.size(0),1,1).to(opt.device)
            data_i = t.bmm(weight,input_).view(input.size(0),int(opt.CR[0]*1024))

            optimizer.zero_grad()
            score = model(data_i)

            loss = criterion(score,target)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
        if(epoch%10==0):
            print("loss value is:",loss_meter.value()[0])
            val_psnr_ave = val(model,val_dataloader)
            print("result on validation dataset is:",val_psnr_ave)

        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
        previous_loss = loss_meter.value()[0]

    model.save()
        

@t.no_grad()
def val(model,dataloader):
    model.eval()
    utils_eval = utils.Evaluation()
    for ii, data in tqdm(enumerate(dataloader)):
        val_input = data[0].float().to(opt.device).squeeze(1)
        val_input_ = val_input.view(val_input.size(0),val_input.size(1)*val_input.size(2),1)
        weight = key_bernoulli_weights.repeat(val_input.size(0),1,1).to(opt.device)
        data_i = t.bmm(weight,val_input_).view(val_input.size(0),int(opt.CR[0]*1024))
        score = model(data_i).unsqueeze(1)
        utils_eval.add(score,val_input.unsqueeze(1).float())
    model.train()
    psnr_ave = utils_eval.psnr_value()
    return psnr_ave

if __name__=='__main__':
    import fire
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    '''
    key_bernoulli_weights = t.FloatTensor(int(opt.CR[0]*1024),1024).bernoulli_(50/100)
    n = 10
    stdv = 1./math.sqrt(n)
    weights_zero = key_bernoulli_weights[key_bernoulli_weights==0].uniform_(-stdv,0)
    weights_one = key_bernoulli_weights[key_bernoulli_weights==1].uniform_(0,stdv)
    key_bernoulli_weights[key_bernoulli_weights==0]=weights_zero
    key_bernoulli_weights[key_bernoulli_weights==1]=weights_one
    key_bernoulli_weights.clamp(-1.0,1.0)
    key_bernoulli_weights = 0.5*(key_bernoulli_weights.sign()+1)
    key_bernoulli_weights[key_bernoulli_weights==0.5]==1
    np.savetxt('key_weight.txt',key_bernoulli_weights)
    '''
    key_bernoulli_weights = np.loadtxt('key_weight.txt')
    key_bernoulli_weights = t.from_numpy(key_bernoulli_weights).float()
    fire.Fire()
