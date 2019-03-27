import os,xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

def readxml(path):
    gtboxes=[]
    tree=ET.parse(path)
    root=tree.getroot()
    image_name = tree.find('filename').text
    for obj in root.iter('object'):
        try:
            bnd=obj.find('bndbox')
            xmin=int(bnd.find('xmin').text)
            ymin=int(bnd.find('ymin').text)
            xmax=int(bnd.find('xmax').text)
            ymax=int(bnd.find('ymax').text)
            gtboxes.append((xmin,ymin,xmax,ymax))
        except Exception as e:
            raise Exception(e+" when {} loading xml file, there are errors".format(path))
    return np.array(gtboxes),image_name


def generate_anchors(featuremap_size=(32,16),scale=16,kmeans=False,k=None):
    """

    :param featuremap_size: 对于vgg16 feature map是/16 /16 .对于其他是不一样的
    :param scale:
    :return:
    """
    if kmeans==False:
        heights=[2]*10
        widths=[1, 2, 4,6,8,12,16,26,32,50]

        heights=np.array(heights).reshape(len(heights),1)
        widths=np.array(widths).reshape(len(widths),1)

        base_anchor=np.array([0,0,1,1])
        xt=(base_anchor[0]+base_anchor[2])*0.5
        yt=(base_anchor[1]+base_anchor[3])*0.5
        x1=xt-widths*0.5
        y1=yt-heights*0.5
        x2=xt+widths*0.5
        y2=yt+heights*0.5
        base_anchor=np.hstack((x1,y1,x2,y2))

        h,w=featuremap_size
        shift_x=np.arange(0,w)*scale
        shift_y=np.arange(0,h)*scale
        anchor=[]
        for i in shift_y:
            for j in shift_x:
                anchor.append(base_anchor+[j,i,j,i])
        return np.array(anchor).reshape(-1,4)
    else:
        """
        使用kmeans聚类生成最合适的k个anchor,参考yolo_v3
        """
        pass
def cal_iou(box1, box1_area, boxes2, boxes2_area):
    """
    box1 [x1,y1,x2,y2]
    boxes2 [Msample,x1,y1,x2,y2]
    """
    x1 = np.maximum(box1[0], boxes2[:, 0])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    y2 = np.minimum(box1[3], boxes2[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
    return iou

def cal_overlaps(boxes1,boxes2):
    """

    :param boxes1: box list anchors
    :param boxes2: box list ground truth box
    :return:
    """
    area1=(boxes1[:,0]-boxes1[:,2])*(boxes1[:,1]-boxes1[:,3])
    area2 = (boxes2[:, 0] - boxes2[:, 2]) * (boxes2[:, 1] - boxes2[:, 3])
    overlaps=np.zeros((boxes1.shape[0],boxes2.shape[0]))
    for i in range(boxes1.shape[0]):
        overlaps[i][:]=cal_iou(boxes1[i],area1[i],boxes2,area2)
    return overlaps

def cal_rpn(imgsize,featuresize,scale,gtboxes,iou_positive,iou_negative,rpn_positive_num,rpn_total_num):
    #使用PIL,img size 是（w,h)
    imgw,imgh=imgsize
    base_anchors=generate_anchors(featuresize,scale)
    overlaps=cal_overlaps(base_anchors,gtboxes)
    #0 represents negative box,1 represents positive box
    labels=np.empty(base_anchors.shape[0])
    labels.fill(-1)
    gt_argmax_overlaps=overlaps.argmax(axis=0)
    anchor_argmax_overlaps=overlaps.argmax(axis=1)
    anchor_max_overlas=overlaps[range(overlaps.shape[0]),anchor_argmax_overlaps]
    labels[anchor_max_overlas>iou_positive]=1
    labels[anchor_max_overlas<iou_negative]=0
    labels[gt_argmax_overlaps]=1

    outside_anchor=np.where((base_anchors[:,0]<0)|(base_anchors[:,1]<0)|
                            (base_anchors[:,2]>=imgw)|(base_anchors[:,3]>=imgh))[0]
    labels[outside_anchor]=-1

    fg_index=np.where(labels==1)[0]
    if len(fg_index)>rpn_positive_num:
        labels[np.random.choice(fg_index,len(fg_index)-rpn_positive_num,replace=False)]=-1
    bg_index=np.where(labels==0)[0]
    num_bg=rpn_total_num-np.sum(labels==1)
    if len(bg_index)>num_bg:
        labels[np.random.choice(bg_index,len(bg_index)-num_bg,replace=False)]=-1
    #calculate bbox targets
    bbox_targets=bbox_transform(base_anchors,gtboxes[anchor_argmax_overlaps,:])
    return [labels,bbox_targets],base_anchors

def bbox_transform(anchors,gtboxes):
    """
    compute the relative predicted horizonal(x) coordinates Vc,Vw
    becuase we have fixed height 16 or 32, width is variable
    :param anchors:
    :param gtboxes:
    :return:
    """
    regr=np.zeros((anchors.shape[0],2))
    Cx=(gtboxes[:,0]+gtboxes[:,2])*0.5
    # a means anchor
    Cxa=(anchors[:,0]+anchors[:,2])*0.5

    W=gtboxes[:,2]-gtboxes[:,0]+1.0
    Wa=anchors[:,2]-anchors[:,0]+1.0
    Vc=(Cx-Cxa)/Wa
    Vw=np.log(W/Wa)
    return np.vstack((Vc,Vw)).transpose()


class MyDataSet(Dataset):
    def __init__(self,image_dir,label_dir,image_shape=None,base_model_name='vgg16',iou_positive=0.7,iou_negative=0.3,
                 iou_select=0.7,rpn_positive_num=150,rpn_total_num=300):
        """

        :param image_dir:
        :param label_dir: xml dir
        :param image_shape:
        """
        if os.path.isdir(image_dir)==False:
            raise Exception('{} not exist'.format(image_dir))
        if os.path.isdir(label_dir)==False:
            raise Exception('{} not exist'.format(label_dir))
        self.image_dir=image_dir
        self.label_dir=label_dir
        if len(os.listdir(self.image_dir))!=len(os.listdir(self.label_dir)):
            raise Exception('image number != label number ')
        else:
            print('image number={}, label number={}'.format(len(os.listdir(self.image_dir)),len(os.listdir(self.label_dir))))
        self.image_shape=image_shape
        self.image_list=os.listdir(self.image_dir)
        self.label_list=os.listdir(self.label_dir)
        self.base_model_name=base_model_name
        self.iou_positive=iou_positive
        self.iou_negative=iou_negative
        self.iou_select=iou_select
        self.rpn_total_num=rpn_total_num
        self.rpn_positive_num=rpn_positive_num

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self,index):
        image_name=self.image_list[index]
        image_path=os.path.join(self.image_dir,image_name)
        label_path=os.path.join(self.label_dir,image_name.replace('.jpg','.xml'))
        gtboxes,xml_filename=readxml(label_path)
        if xml_filename!=image_name:
            raise Exception('read xml error')
        img=Image.open(image_path)
        """
        修改原始图像后，对应的box也要修改
        TODO!
        """
        if self.image_shape!=None:
            #image_shape=(width,height)
            original_size=img.size
            x_scale = self.image_shape[0] / original_size[0]
            y_scale = self.image_shape[1] / original_size[1]
            newbox = []
            for i in range(len(gtboxes)):
                newbox.append(
                    [gtboxes[i][0] * x_scale, gtboxes[i][1] * y_scale, gtboxes[i][2] * x_scale, gtboxes[i][3] * y_scale]
                )
            img=img.resize((self.image_shape[0],self.image_shape[1]),Image.ANTIALIAS)
            gtboxes=np.array(newbox)
        w,h=img.size
        if self.base_model_name=='vgg16':
            scale=16
        else:
            scale=32
        [cls,regr],_=cal_rpn(imgsize=(w,h),featuresize=(int(h/scale),int(w/scale)),scale=scale,gtboxes=gtboxes,
                             iou_positive=self.iou_positive,iou_negative=self.iou_negative,
                             rpn_total_num=self.rpn_total_num,rpn_positive_num=self.rpn_positive_num)
        img=np.array(img)
        img=(img / 255.0) * 2.0 - 1.0
        regr=np.hstack([cls.reshape(cls.shape[0],1),regr])
        cls=np.expand_dims(cls,axis=0)
        #pytorch 是channel first的,所以得把numpy 图片的第三维channel放在首位
        img=torch.from_numpy(img.transpose([2,0,1])).float()
        cls=torch.from_numpy(cls).float()
        regr=torch.from_numpy(regr).float()
        return img.cpu(),cls.cpu(),regr.cpu()

if __name__=='__main__':
    # print(readxml('D:\py_projects\data_new\data_new\data\\annotation\img_calligraphy_00001_bg.xml'))
    # img = Image.open('D:\py_projects\data_new\data_new\data\\train_img\img_calligraphy_00001_bg.jpg')
    # img=(np.array(img) / 255.0) * 2.0 - 1.0
    # print(img.shape)
    # print(img)
    #(w,h)
    a=generate_anchors()
    for i in a[:,0]:
        print(i)
    print(a.shape)




