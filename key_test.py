from config import opt
import os
import torch as t
import models
import utils
import numpy as np
import math
from torchvision import transforms as tf
import cv2

@t.no_grad()
def test(**kwargs):
    opt._parse(kwargs)
    model = models.Key_CSVideoNet(1,opt.Height,opt.Width,1)
    if opt.load_model_path:
        model.load(opt.load_model_path)
        print("model load success!")
    #if t.cuda.device_count()>1:
    #    model = t.nn.DataParallel(model,device_ids=[0,1])
    model.to(opt.device)
    model.eval()
    
    videos = [os.path.join(opt.test_data_root,video) for video in os.listdir(opt.test_data_root)]
    video_num = len(videos)
    
 
    for item in videos:
        ue = utils.Evaluation()
        uv = utils.Video(item,opt.save_test_root,1,opt.overlap,opt.Height)
        frames = uv.frame_capture()
        input = uv.frame_unfold(frames).float()
        print(input.size())
        frames_num = input.size(0)
        block_num_h = input.size(1)
        block_num_w = input.size(2)
        output_ = t.zeros(frames_num,block_num_h,block_num_w,opt.Height,opt.Width)
        input_ = input.view(frames_num,block_num_h*block_num_w,opt.Height*opt.Width).unsqueeze(3).to(opt.device)
        for i in range(frames_num):
            weight = key_bernoulli_weights.repeat(input_.size(1),1,1).to(opt.device)
            #print(weight.size())
            #print(input_.size())
            input_t = input_[i]/255.0
            data_i = t.bmm(weight,input_t).view(input_.size(1),1024)
            output = model(data_i)
            ue.add(output.unsqueeze_(0),input_t.view(input_.size(1),opt.Height,opt.Width).unsqueeze_(0))
            output_[i] = (output.view(block_num_h,block_num_w,opt.Height,opt.Width))*(255.0/t.max(output))
        frames_output = uv.frame_fold(output_)
        uv.generate_video(frames_output)
        print("the average PSNR is",ue.psnr_value())
    '''
    test_data = UCF101(opt.test_data_root,train=True)
    test_dataloader = DataLoader(test_data,opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    for ii,data in tqdm(enumerate(test_dataloader)):
        input = data[0].float().to(opt.device).squeeze_(1)   
        input_ = input.view(input.size(0),input.size(1)*input.size(2),1)
        weight = key_bernoulli_weights.repeat(input.size(0),1,1).to(opt.device)
        data_i = t.bmm(weight,input_).view(input.size(0),1024)
    '''
if __name__=='__main__':
    import fire
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    key_bernoulli_weights = np.loadtxt('key_weight.txt')
    key_bernoulli_weights = t.from_numpy(key_bernoulli_weights).float()
    fire.Fire()
    
