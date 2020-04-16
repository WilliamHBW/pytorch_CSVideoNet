from config import opt
import os
import torch as t 
import models
import utils
import numpy as np
import cv2

@t.no_grad()
def test(**kwargs):
    opt._parse(kwargs)

    model = models.CSVideoNet(opt.CR,opt.seqLength,opt.Height,opt.Width,1)
    if opt.load_model_path:
        model.load(opt.load_model_path,False)
    model.to(opt.device)

    videos = [os.path.join(opt.test_data_root,video) for video in os.listdir(opt.test_data_root)]
    video_num = len(videos)
    
    for item in videos:
        ue = utils.Evaluation()
        uv = utils.Video(item,opt.save_test_root,1,opt.overlap,opt.Height)
        frames = uv.frame_capture()
        input = uv.frame_unfold(frames)
        print(input.size())
        frames_num = input.size(0)
        block_num_h = input.size(1)
        block_num_w = input.size(2)
        result_b = t.zeros(frames_num,block_num_h,block_num_w,opt.Height,opt.Width)
        if frames_num%opt.seqLength != 0:
            frames_num = frames_num-(frames_num%opt.seqLength)
            print("Warning: input video length has been changed.")
        input_ = input[0:frames_num,:,:,:,:]
        frame_idx = 0
        for frames_item in [input_[i:i+opt.seqLength] for i in range(0,frames_num,opt.seqLength)]:
            for j in range(block_num_h):
                for k in range(block_num_w):
                    frames_input_b = ((frames_item[:,j,k,:,:].float())/255.0).to(opt.device)
                    #print("input max:",t.max(frames_input_b))
                    key_weight = key_bernoulli_weights.to(opt.device).float()
                    non_key_weight = non_key_bernoulli_weights.to(opt.device).float()
                    key_m = t.mm(key_weight,(frames_input_b[0]).view(opt.Height*opt.Width,1)).squeeze_(1).unsqueeze_(0)
                    nonkey_m = t.mm(non_key_weight,(frames_input_b[1]).view(opt.Height*opt.Width,1)).squeeze_(1).unsqueeze_(0).unsqueeze_(0)
                    frames_output_b = model(key_m,nonkey_m)
                    if(t.max(frames_output_b)>1.0):
                        print("output max:",t.max(frames_output_b))
                    result_b[frame_idx:frame_idx+opt.seqLength,j,k,:,:] = frames_output_b*255.0
                    #print(t.max(frames_output_b),t.max(frames_input_b))
                    ue.add(frames_output_b,frames_input_b.unsqueeze_(0))
            frame_idx = frame_idx + opt.seqLength
        #print(frames_input_b.size(),frames_output_b.size())
        #data_write1 = (frames_input_b[:,0,:,:]*255.0).squeeze_(0).unsqueeze_(2)
        #print(data_write1.size())
        #data_write2 = (frames_output_b[:,0,:,:]*255.0).squeeze_(0).unsqueeze_(2)
        #print(data_write2.size())
        #cv2.imwrite("./result/2.png",data_write1.cpu().numpy())
        #cv2.imwrite("./result/3.png",data_write2.cpu().numpy())
        frames_output = uv.frame_fold(result_b)
        uv.generate_video(frames_output)
        print("the average PSNR is ",ue.psnr_value())
        
def help():
    """
    打印帮助的信息： python file.py help
    """
    
    print("""
    usage : python file.py <function> [--args=value]
    <function> := | test | help
    example: 
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    key_bernoulli_weights = np.loadtxt('key_weight.txt')
    key_bernoulli_weights = t.from_numpy(key_bernoulli_weights).float()
    non_key_bernoulli_weights = np.loadtxt('nonkey_weight.txt')
    non_key_bernoulli_weights = t.from_numpy(non_key_bernoulli_weights).float() 
    import fire
    fire.Fire()
