from config import opt
import os 
import torch as t 
import models 
import utils 
from data.dataset_csvideonet import UCF101 
from torch.utils .data import DataLoader
from torchnet import meter 
from tqdm import tqdm 

def train(**kwargs):
    opt._parse(kwargs)

    model = models.Key_CSVideoNet(opt.CR[0],opt.Height,opt.Width,1)
    #if t.cuda.device_count() > 1:
    #    model = t.nn.DataParallel(model,device_ids = [0,1,2])
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
            for seq_num in range(opt.seqLength):
                model.Measurements.binarization()

                input = data[:,seq_num,:,:].float().to(opt.device)
                target = input

                optimizer.zero_grad()
                score = model(input)

                loss = criterion(score,target)
                loss.backward()
                optimizer.step()

                model.Measurements.restore()

                loss_meter.add(loss.item())

        model.save()

        val_psnr_ave = val(model,val_dataloader)
        print("result on validation dataset is:",val_psnr_ave)

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
        for seq_num in range(opt.seqLength):
            val_input = data[:,seq_num,:,:].float().to(opt.device)
            score = model(val_input).unsqueeze(1)
            utils_eval.add(score,val_input.unsqueeze(1).float())
    model.train()
    psnr_ave = utils_eval.psnr_value()
    return psnr_ave

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'
    import fire
    fire.Fire()
