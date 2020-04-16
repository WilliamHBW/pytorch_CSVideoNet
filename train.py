from config import opt
import os
import torch as t 
import models
import utils
from data.dataset_csvideonet import UCF101
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
import math
import numpy as np

def train(**kwargs):
    opt._parse(kwargs)
    
    model = models.CSVideoNet(opt.CR,opt.seqLength,opt.Height,opt.Width,1)
    if opt.load_model_path:
        model.load(opt.load_model_path,True)
    '''
    if t.cuda.device_count() > 1:
        model = t.nn.DataParallel(model,device_ids=[0,1])
    '''
    model.to(opt.device)

    train_data = UCF101(opt.train_data_root,train=True)
    val_data = UCF101(opt.train_data_root,train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    
    criterion = t.nn.MSELoss()
    lr = opt.lr
    optimizer = t.optim.SGD(model.parameters(),
                           lr = lr,
                           momentum=opt.momentum)

    utils_eval = utils.Evaluation()
    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e10

    for epoch in range(opt.max_epoch):

        loss_meter.reset()

        for ii,data in tqdm(enumerate(train_dataloader)):
            #train model
            input = data
            target = t.zeros(data[0].size(0),opt.seqLength,32,32).to(opt.device)
            nonkey_m = t.zeros(input[0].size(0),opt.seqLength-1,int(1024*opt.CR[1])).to(opt.device)
            for i in range(opt.seqLength):
                target[:,i,:,:] = data[i].float().to(opt.device).squeeze(1)
            key_weight = key_bernoulli_weights.repeat(input[0].size(0),1,1).to(opt.device)
            non_key_weight = non_key_bernoulli_weights.repeat(input[1].size(0),1,1).to(opt.device)
            key_m = t.bmm(key_weight,input[0].float().to(opt.device).view(input[0].size(0),1024,1)).view(input[0].size(0),int(1024*opt.CR[0]))
            nonkey_m[:,0,:] = t.bmm(non_key_weight,input[1].view(input[1].size(0),1024,1).float().to(opt.device)).view(input[1].size(0),int(1024*opt.CR[1]))
            #nonkey_m[:,1,:] = t.bmm(non_key_weight,input[2].view(input[2].size(0),1024,1).float().to(opt.device)).view(input[2].size(0),int(1024*opt.CR[1]))
            #nonkey_m[:,2,:] = t.bmm(non_key_weight,input[3].view(opt.batch_size,1024,1).float().to(opt.device)).view(opt.batch_size,int(1024*opt.CR[1]))
            #nonkey_m[:,3,:] = t.bmm(non_key_weight,input[4].view(opt.batch_size,1024,1).float().to(opt.device)).view(opt.batch_size,int(1024*opt.CR[1]))
            #nonkey_m[:,4,:] = t.bmm(non_key_weight,input[5].view(opt.batch_size,1024,1).float().to(opt.device)).view(opt.batch_size,int(1024*opt.CR[1]))
            #nonkey_m[:,5,:] = t.bmm(non_key_weight,input[6].view(opt.batch_size,1024,1).float().to(opt.device)).view(opt.batch_size,int(1024*opt.CR[1]))
            #nonkey_m[:,6,:] = t.bmm(non_key_weight,input[7].view(opt.batch_size,1024,1).float().to(opt.device)).view(opt.batch_size,int(1024*opt.CR[1]))
            #nonkey_m[:,7,:] = t.bmm(non_key_weight,input[8].view(opt.batch_size,1024,1).float().to(opt.device)).view(opt.batch_size,int(1024*opt.CR[1]))
            #nonkey_m[:,8,:] = t.bmm(non_key_weight,input[9].view(opt.batch_size,1024,1).float().to(opt.device)).view(opt.batch_size,int(1024*opt.CR[1]))

            optimizer.zero_grad()
            score = model(key_m,nonkey_m)

            loss = criterion(score,target)
            loss.backward()
            optimizer.step()

            #t.nn.utils.clip_grad_norm(model.modules.parameters(),opt.gradient_clipping)

            loss_meter.add(loss.item())
       
        val_psnr_ave = val(model,val_dataloader)
        print(epoch,"result on validation dataset is:",val_psnr_ave)

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
        input = data
        target = t.zeros(data[0].size(0),opt.seqLength,32,32).to(opt.device)
        nonkey_m = t.zeros(input[0].size(0),opt.seqLength-1,int(1024*opt.CR[1])).to(opt.device)
        for i in range(opt.seqLength):
            target[:,i,:,:] = data[i].float().to(opt.device).squeeze(1)
        key_weight = key_bernoulli_weights.repeat(input[0].size(0),1,1).to(opt.device)
        non_key_weight = non_key_bernoulli_weights.repeat(input[1].size(0),1,1).to(opt.device)
        key_m = t.bmm(key_weight,input[0].float().to(opt.device).view(input[0].size(0),1024,1)).view(input[0].size(0),int(1024*opt.CR[0]))
        nonkey_m[:,0,:] = t.bmm(non_key_weight,input[1].view(input[1].size(0),1024,1).float().to(opt.device)).view(input[1].size(0),int(1024*opt.CR[1]))
        #nonkey_m[:,1,:] = t.bmm(non_key_weight,input[2].view(input[2].size(0),1024,1).float().to(opt.device)).view(input[2].size(0),int(1024*opt.CR[1]))
        #nonkey_m[:,2,:] = t.bmm(non_key_weight,input[3].view(opt.batch_size,1024,1).float().to(opt.device)).view(opt.batch_size,int(1024*opt.CR[1]))
        #nonkey_m[:,3,:] = t.bmm(non_key_weight,input[4].view(opt.batch_size,1024,1).float().to(opt.device)).view(opt.batch_size,int(1024*opt.CR[1]))
        #nonkey_m[:,4,:] = t.bmm(non_key_weight,input[5].view(opt.batch_size,1024,1).float().to(opt.device)).view(opt.batch_size,int(1024*opt.CR[1]))
        #nonkey_m[:,5,:] = t.bmm(non_key_weight,input[6].view(opt.batch_size,1024,1).float().to(opt.device)).view(opt.batch_size,int(1024*opt.CR[1]))
        #nonkey_m[:,6,:] = t.bmm(non_key_weight,input[7].view(opt.batch_size,1024,1).float().to(opt.device)).view(opt.batch_size,int(1024*opt.CR[1]))
        #nonkey_m[:,7,:] = t.bmm(non_key_weight,input[8].view(opt.batch_size,1024,1).float().to(opt.device)).view(opt.batch_size,int(1024*opt.CR[1]))
        #nonkey_m[:,8,:] = t.bmm(non_key_weight,input[9].view(opt.batch_size,1024,1).float().to(opt.device)).view(opt.batch_size,int(1024*opt.CR[1]))
        score = model(key_m,nonkey_m)
        utils_eval.add(score,target)

    model.train()
    psnr_ave = utils_eval.psnr_value()
    return psnr_ave

def help():
    """
    打印帮助的信息： python file.py help
    """
    
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    key_bernoulli_weights = np.loadtxt('key_weight.txt')
    key_bernoulli_weights = t.from_numpy(key_bernoulli_weights).float() 
    n = 10
    stdv = 1./math.sqrt(n)
    '''
    non_key_bernoulli_weights = t.FloatTensor(int(1024*opt.CR[1]),1024).bernoulli_(50/100)
    nk_weights_zero = non_key_bernoulli_weights[non_key_bernoulli_weights==0].uniform_(-stdv,0)
    nk_weights_one = non_key_bernoulli_weights[non_key_bernoulli_weights==1].uniform_(0,stdv)
    non_key_bernoulli_weights[non_key_bernoulli_weights==0]=nk_weights_zero
    non_key_bernoulli_weights[non_key_bernoulli_weights==1]=nk_weights_one
    non_key_bernoulli_weights.clamp(-1.0,1.0)
    non_key_bernoulli_weights = 0.5*(non_key_bernoulli_weights.sign()+1)
    non_key_bernoulli_weights[non_key_bernoulli_weights==0.5]==1
    np.savetxt('nonkey_weight.txt',non_key_bernoulli_weights)
    '''
    non_key_bernoulli_weights = np.loadtxt('nonkey_weight.txt')
    non_key_bernoulli_weights = t.from_numpy(non_key_bernoulli_weights).float()
    import fire
    fire.Fire()
