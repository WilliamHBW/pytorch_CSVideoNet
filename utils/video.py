import cv2
import os
from skimage import io 
import numpy as np 
import torch as t

class Video(object):
    def __init__(self,loadpath,saveroot,time_freq,overlap,block):
        self.loadpath = loadpath
        self.saveroot = saveroot
        self.time_freq = time_freq
        self.overlap = overlap
        self.block = block
        self.frame_h = None
        self.frame_w = None
        self.input_patch_numel = None
        self.input_patch_size = None
        self.input_frame_numel = None
        self.input_frame_size = None

        if not os.path.exists(saveroot):
            os.makedirs(saveroot)
    
    def frame_capture(self):
        #output data type list(numpy array)
        #output data shape(frame_num height width)
        img_idx = 1
        videopath = self.loadpath
        frame_imgs = []
        output = []
        print("video loadpath is",videopath)

        vc = cv2.VideoCapture(videopath)

        video_frame_idx = 1

        if vc.isOpened():
            rval,frame = vc.read()
        else:
            rval = False
            
        while rval:
            rval,frame = vc.read()
            if(video_frame_idx%self.time_freq == 0):
                frame_imgs.append(frame)
                img_idx = img_idx + 1
            video_frame_idx = video_frame_idx + 1
            cv2.waitKey(1)
        vc.release()
        frame_imgs.pop()
        for frame_item in frame_imgs:
            frame_item_ = cv2.cvtColor(frame_item,cv2.COLOR_BGR2GRAY)
            f_h,f_w = frame_item_.shape
            frame_item_ = frame_item_[int(f_h/2)-80:int(f_h/2)+80,int(f_w/2)-80:int(f_w/2)+80]
            output.append(frame_item_)
        return output
    
    def frame_unfold(self,frame_imgs):
        #input data type list(numpy array)
        #input data shape [frame_num,height,width]
        #output data type torch tensor
        #output data size [frame_num,block_num_h,block_num_w,block_width,block_width]
        frame_imgs_n = np.array(frame_imgs)
        frame_imgs_t = t.from_numpy(frame_imgs_n)
        self.input_frame_size = frame_imgs_t.size()
        self.input_frame_numel = frame_imgs_t.numel()
        self.frame_h = frame_imgs_t.size(1)
        self.frame_w = frame_imgs_t.size(2)
        if((self.frame_h-self.block)%(self.block-self.overlap)!=0 or (self.frame_w-self.block)%(self.block-self.overlap)!=0):
            print("frame size is",self.input_frame_size,"error:overlap size mismatch!")
            return 0
        else:
            output_ = frame_imgs_t.unfold(1,self.block,self.block-self.overlap).unfold(2,self.block,self.block-self.overlap)
            output = output_.contiguous()
            self.input_patch_numel = output.numel()
            self.input_patch_size = output.size()
            return output

    def frame_fold(self,input_patches):
        #input data type torch tensor
        #input data shape [frame_num,block_num_h,block_num_w,block_width,block_width]
        #output data type numpy array
        #output data shape [frame_num,height,width]
        input_patches = input_patches.float()
        idx = t.zeros(self.input_frame_numel).long()
        t.arange(0,self.input_frame_numel,out=idx)
        idx = idx.view(self.input_frame_size)
        idx_unfold = idx.unfold(1,self.block,self.block-self.overlap).unfold(2,self.block,self.block-self.overlap)
        idx_unfold = idx_unfold.contiguous().view(-1)

        video = t.zeros(self.input_frame_size).view(-1)
        video_ones = t.zeros(self.input_frame_size).view(-1)
        patches_ones = (t.zeros_like(input_patches)+1).view(-1)

        input_patches = input_patches.contiguous().view(-1)
        video.index_add_(0,idx_unfold,input_patches)
        video_ones.index_add_(0,idx_unfold,patches_ones)
        output = (video/video_ones).view(self.input_frame_size)
        output_ = output.numpy()
        return output_ 


    def generate_video(self,data):
        #input data type numpy array
        #input data shape [frame_num,height,width]
        frame_num = data.shape[0]
        savepath = self.saveroot + "/" + self.loadpath.split('.')[-2].split('/')[-1]+'.mp4'
        print("video savepath is ",savepath)
        videoWriter = cv2.VideoWriter(savepath,cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),30,(self.frame_w,self.frame_h))
        for i in range(frame_num):
            # load pictures from your path	
            img_ = data[i].astype(np.uint8)
            img  = cv2.cvtColor(img_,cv2.COLOR_GRAY2BGR)
            videoWriter.write(img)
        videoWriter.release()

'''
loadpath = "/Users/willamhuang/Desktop/1/1.avi"
savepath = "/Users/willamhuang/Desktop/2"
video_test = Video(loadpath,savepath,1,0,32)
a = video_test.frame_capture()
print(len(a))
print(a[0].shape)
b = video_test.frame_unfold(a)
print(b.size())
c = video_test.frame_fold(b)
print(c.shape)
#cv2.imwrite('/Users/willamhuang/Desktop/1/6.jpg',c[300])
video_test.generate_video(c)
'''