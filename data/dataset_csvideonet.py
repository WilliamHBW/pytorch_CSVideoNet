import torch as t 
from torch.utils import data
import numpy as np 
from torchvision import transforms as tf 
import os 
from operator import attrgetter
import cv2
from config import opt
from PIL import Image

class filename(object):
    def __init__(self,name,g,c,f,b):
        self.name = name
        self.g = g 
        self.c = c
        self.f = f
        self.b = b
    def __repr(self):
        return repr((self.name,self.g,self.c,self.f,self.b))
    def get_filename(self):
        return self.name+'_g'+self.g+'_c'+self.c+'_'+self.f+'_'+self.b

class UCF101(data.Dataset):
    def __init__(self,root,transforms=None,train=True):
        imgs = [os.path.join(root,img) for img in os.listdir(root)]
        imgs_num = len(imgs)
        self.n = 10
        self.root = root
        self.train = train

        #train: data/train/v_YOYO_g01_c01_1_1.jpg
        train_file = []
        for item in imgs:
            name = item.split('.')[-2].split('_')[-5].split('/')[-1]
            g = item.split('.')[-2].split('_')[-4].split('g')[-1]
            c = item.split('.')[-2].split('_')[-3].split('c')[-1]
            f = item.split('.')[-2].split('_')[-2]
            b = item.split('.')[-2].split('_')[-1]
            filename_ = filename(name,g,c,f,b)
            train_file.append(filename_)
        train_file.sort(key=attrgetter('name','g','c'))
        train_file.sort(key=lambda x:int(x.b))

        #use part data as train dataset and others as val dataset
        if self.train:
            train_dataset_ = train_file[0:int(0.7*imgs_num)]
            #train_dataset_ = train_file
        else:
            train_dataset_ = train_file[int(0.7*imgs_num):]
        one_video_train_dataset = []
        self.train_dataset = []
        f_name = train_dataset_[0].name
        f_g = train_dataset_[0].g
        f_c = train_dataset_[0].c
        f_b = train_dataset_[0].b
        for item in train_dataset_:
            if(item.name==f_name and item.g==f_g and item.c==f_c and item.b==f_b):
                one_video_train_dataset.append(item)
            else:
                one_video_train_dataset.sort(key=lambda x:int(x.f))
                tmp_num = len(one_video_train_dataset)
                for item_1 in [one_video_train_dataset[i:i+self.n] for i in range(0,tmp_num,self.n)]:
                    if(len(item_1)==10):
                        self.train_dataset.append(item_1)
                    else:
                        break
                    #for i in range(10):
                     #   print(item_1[i].get_filename)
                one_video_train_dataset = []
                one_video_train_dataset.append(item)
            f_name = item.name
            f_g = item.g
            f_c = item.c
            f_b = item.b

        #do random transforms
        
        if transforms is None:
            normalize = tf.Normalize(mean=[0.485],std=[0.229])
            self.transforms = tf.Compose([
                    tf.ToTensor(),
                    normalize
                ])
        

    def __getitem__(self,index):
        #output data type torch tensor
        #output data shape [10,height,width]
        output_ = []
        img_path_object = self.train_dataset[index]
        for i in range(self.n):
            img_path = self.root+'/'+img_path_object[i].name + '_g'+ img_path_object[i].g+'_c'+img_path_object[i].c+'_'+ img_path_object[i].f + '_' + img_path_object[i].b + '.jpg'
            data = Image.open(img_path)
            data = self.transforms(data)
            output_.append(data)
        return output_
    
    def __len__(self):
        return len(self.train_dataset)

'''
saveroot = "/Users/willamhuang/Desktop/1/train"
cs_dataset = UCF101(saveroot)
print(cs_dataset.__len__())
print(len(cs_dataset.train_dataset[1]))
print(cs_dataset.__getitem__(1).size())
'''
