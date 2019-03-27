import numpy as np
import os
from PIL import Image
from Detection.CTPN_vertical.utils import cal_rpn
class DataGenerator(object):
    def __init__(self,image_dir, mode,label_dir,val_size=2000,image_shape=None,k=10, base_model_name='vgg16',
                 iou_positive=0.7, iou_negative=0.3, iou_select=0.7, rpn_positive_num=150, rpn_total_num=300,train_bs=50,val_bs=50
                 ):
        self.mode=mode
        assert self.mode in ['train','val']
        self.image_dir=image_dir
        self.label_dir=label_dir
        if os.path.isdir(self.image_dir)==False or os.path.isdir(self.label_dir)==False:
            raise Exception('valid dir')
        if len(os.listdir(self.image_dir))!=len(os.listdir(self.label_dir)):
            raise Exception('Not equalled lengths of dir file ')
        self.val_size=val_size
        self.image_shape=image_shape
        self.k=k
        self.base_model_name=base_model_name
        self.iou_positive=iou_positive
        self.iou_negative=iou_negative
        self.iou_select=iou_select
        self.rpn_positive_num=rpn_positive_num
        self.rpn_total_num=rpn_total_num
        self.train_bs=train_bs
        self.val_bs=val_bs
        self.__shuffle__()
    def __shuffle__(self):
        self.total_len=len(os.listdir(self.image_dir))
        self.all_index=np.array(range(self.total_len))
        np.random.shuffle(self.all_index)
        self.val_index=np.random.choice(self.all_index,size=self.val_size,replace=False)
        self.train_index=[i for i in self.all_index if i not in self.val_index]
    def generator(self):
        input_imgs=[]
        cls=[]
        regr=[]
        count = 0
        mapping={'train':self.train_index,'val':self.val_index}
        img_index = mapping[self.mode]
        batch_size_mapping={'train':self.train_bs,'val':self.val_bs}


if __name__=='__main__':
    d=DataGenerator(image_dir='D:\py_projects\data_new\data_new\data\\train_img',
                    label_dir='D:\py_projects\data_new\data_new\data\\annotation',
                    mode='train')













