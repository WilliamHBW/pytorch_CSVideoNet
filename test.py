from config import opt
import os
import torch as t 
import models
import utils
import numpy as np

@t.no_grad()
def test(**kwargs):
    opt._parse(kwargs)

    model = models.CSVideoNet(opt.CR,opt.seqLength,opt.Height,opt.Width,1)
    if opt.load_model_path:
        model.load(opt.load_model_path[0],opt.load_model_path[1],opt.load_model_path[2],opt.load_model_path[3])
    model.to(opt.device)

    videos = [os.path.join(opt.test_data_root,video) for video in os.listdir(opt.test_data_root)]
    video_num = len(videos)
    
    for item in videos:
        ue = utils.Evaluation()
        uv = utils.Video(item,opt.save_test_root,1,opt.overlap,opt.Height)
        frames = uv.frame_capture()
        input = uv.frame_unfold(frames)
        frames_num = input.size(0)
        block_num_h = input.size(1)
        block_num_w = input.size(2)
        result_b = t.zeros(frames_num,block_num_h,block_num_w,opt.Height,opt.Width).to(opt.device)
        if frames_num%10 != 0:
            frames_num = frames_num-(frames_num%10)
            print("Warning: input video length has been changed.")
        input_ = input[0:frames_num,:,:,:,:]
        frame_idx = 0
        for frames_item in [input_[i:i+opt.seqLength] for i in range(0,frames_num,opt.seqLength)]:
            for j in range(block_num_h):
                for k in range(block_num_w):
                    frames_input_b = frames_item[:,j,k,:,:].unsqueeze(0).float()
                    frames_output_b = model(frames_input_b)
                    result_b[frame_idx:frame_idx+opt.seqLength,j,k,:,:] = frames_output_b
                    ue.add(frames_output_b,frames_input_b)
            frame_idx = frame_idx + 10
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
    import fire
    fire.Fire()
