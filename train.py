from config import opt
import os
import torch as t 
import models
import utils
from data.dataset_csvideonet import UCF101
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm

def train(**kwargs):
    opt._parse(kwargs)
    
    model = models.CSVideoNet(opt.CR,opt.seqLength,opt.Height,opt.Width,1)
    if opt.load_model_path:
        model.load(opt.load_model_path[0],opt.load_model_path[1],opt.load_model_path[2],opt.load_model_path[3])
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
            #train model
            model.measurement.binarization()

            input = data.float().to(opt.device)
            target = data.float().to(opt.device)

            optimizer.zero_grad()
            score = model(input)

            loss = criterion(score,target)
            loss.backward()
            optimizer.step()

            model.measurement.restore()
            #t.nn.utils.clip_grad_norm(model.modules.parameters(),opt.gradient_clipping)

            loss_meter.add(loss.item())
    
        model.save()

        val_psnr_ave = val(model,val_dataloader)
        print(epoch,"result on validation dataset is:",val_psnr_ave)

        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
        previous_loss = loss_meter.value()[0]

@t.no_grad()
def val(model,dataloader):
    model.eval()
    utils_eval = utils.Evaluation()
    for ii, data in tqdm(enumerate(dataloader)):
        val_input = data.float().to(opt.device)
        score = model(val_input)
        utils_eval.add(score,data.float())

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    import fire
    fire.Fire()
