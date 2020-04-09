import os
import utils
import cv2

rootpath = "/Users/willamhuang/Desktop/video/Archery"
videos = [os.path.join(rootpath,video) for video in os.listdir(rootpath)]
videos_num = len(videos)
saveroot = "/Users/willamhuang/Desktop/1/train"
if not os.path.exists(saveroot):
    os.makedirs(saveroot)
for item in videos:
    print(item.split('.')[-1])
    if(item.split('.')[-1]!='avi'):
        continue
    savename_p1 = item.split('.')[-2].split('_')[-3]+'_'+item.split('.')[-2].split('_')[-2]+'_'+item.split('.')[-2].split('_')[-1]
    print(savename_p1)
    video_test = utils.Video(item,None,1,0,32)
    a = video_test.frame_capture()
    b_ = video_test.frame_unfold(a)
    b = b_.numpy()
    frame_num,block_num_h,block_num_w,block_width,_ = b.shape
    frame_cnt = 0
    for i in range(frame_num):
        block_cnt = 0
        for j in range(block_num_h*block_num_w):
            savename_p2 = str(frame_cnt)+'_'+str(block_cnt)+'.jpg'
            savepath = saveroot+'/'+savename_p1+'_'+savename_p2
            print(savepath)
            img = b[i][int(block_cnt/block_num_w)][int(block_cnt%block_num_w)]
            cv2.imwrite(savepath,img)
            block_cnt = block_cnt+1
        frame_cnt = frame_cnt + 1



