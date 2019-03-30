import numpy as np
import os
from PIL import Image
from Detection.CTPN_vertical.utils import cal_rpn,readxml,generate_anchors,bbox_trasfor_inv,nms
import Detection.CTPN_vertical.CP as cp
from Detection.CTPN_vertical.model_keras import proposal_model
from Detection.CTPN_vertical.utils import drawRect,TextProposalConnectorOriented
from Detection.CTPN_vertical.LineGraphy import clip_boxes
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping,TensorBoard
import cv2
from math import *
def rotate(img_path):
    img=cv2.imread(img_path)
    height, width = img.shape[:2]

    degree = 90
    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))  # 这个公式参考之前内容
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
    matRotation[1, 2] += (heightNew - height) / 2

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

    return Image.fromarray(imgRotation)


def rpn_regr_loss(y_true,y_pred):

    sigma = 9.0
    cls = y_true[0, :, 0]
    regr = y_true[0, :, 1:3]
    regr_keep = tf.where(K.equal(cls, 1))[:, 0]
    regr_true = tf.gather(regr, regr_keep)
    regr_pred = tf.gather(y_pred[0], regr_keep)
    diff = tf.abs(regr_true - regr_pred)
    less_one = tf.cast(tf.less(diff, 1.0 / sigma), 'float32')
    loss = less_one * 0.5 * diff ** 2 * sigma + tf.abs(1 - less_one) * (diff - 0.5 / sigma)
    loss = K.sum(loss, axis=1)
    return K.switch(tf.size(loss) > 0, K.mean(loss), K.constant(0.0))

def rpn_cls_loss(y_true,y_pred):

    y_true = y_true[0][0]
    cls_keep = tf.where(tf.not_equal(y_true, -1))[:, 0]
    cls_true = tf.gather(y_true, cls_keep)
    cls_pred = tf.gather(y_pred[0], cls_keep)
    cls_true = tf.cast(cls_true, 'int64')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cls_true, logits=cls_pred)
    return K.switch(tf.size(loss) > 0, K.clip(K.mean(loss), 0, 10), K.constant(0.0))


class DataGenerator(object):
    def __init__(self,image_dir, mode,label_dir,input_index,batch_size,image_shape=None,k=10, base_model_name='vgg16',
                 iou_positive=0.7, iou_negative=0.3, iou_select=0.7, rpn_positive_num=150, rpn_total_num=300
                 ):
        """
        custom data generator for tensorflow

        :param image_dir: all image folder
        :param mode: 'train' or 'val'
        :param label_dir: annotation folder
        :param input_index: train index or val index
        :param batch_size:
        :param image_shape: reshape all the images at a static shape for batch training
        :param k: k anchors for each 1x1 grid in the feature map
        :param base_model_name:vgg16 resnet50 etc
        :param iou_positive:
        :param iou_negative:
        :param iou_select:
        :param rpn_positive_num:
        :param rpn_total_num:
        """
        self.mode=mode
        assert self.mode in ['train','val']
        self.image_dir=image_dir
        self.label_dir=label_dir
        if os.path.isdir(self.image_dir)==False or os.path.isdir(self.label_dir)==False:
            raise Exception('invalid directory,please check your image dir and label dir')
        if len(os.listdir(self.image_dir))!=len(os.listdir(self.label_dir)):
            raise Exception('Different lengths of input dir file,in fact, they are supposed to have the same file number')
        self.image_shape=image_shape
        self.k=k
        self.base_model_name=base_model_name
        self.iou_positive=iou_positive
        self.iou_negative=iou_negative
        self.iou_select=iou_select
        self.rpn_positive_num=rpn_positive_num
        self.rpn_total_num=rpn_total_num
        self.batch_size=batch_size
        self.input_index=input_index
        if self.image_shape==None:#else (w,h)
            self.batch_size=1
        self.steps_per_epoch=len(self.input_index)//self.batch_size
    def generator(self,turn=False):
        input_batch_imgs=[]
        batch_cls=[]
        batch_regr=[]
        batch_count = 0
        all_image_name=os.listdir(self.image_dir)
        while True:

            for i in self.input_index:
                img_name=all_image_name[i]
                label_name=img_name.replace('.jpg','.xml')
                img=Image.open(os.path.join(self.image_dir,img_name))
                gtboxes, xml_filename = readxml(os.path.join(self.label_dir,label_name))
                if xml_filename!=img_name:
                    raise Exception('read xml error')
                if turn:
                    ow,oh=img.size
                    img=rotate(os.path.join(self.image_dir,img_name))
                    newbox=[]
                    for i in gtboxes:
                        newbox.append([i[1], ow - i[2], i[3], ow - i[0]])
                    gtboxes=np.array(newbox)

                if self.image_shape!=None:
                   if turn==False:
                       original_size = img.size
                       x_scale = self.image_shape[0] / original_size[0]
                       y_scale = self.image_shape[1] / original_size[1]
                       newbox = []
                       for i in range(len(gtboxes)):
                           newbox.append(
                               [gtboxes[i][0] * x_scale, gtboxes[i][1] * y_scale, gtboxes[i][2] * x_scale,
                                gtboxes[i][3] * y_scale]
                           )
                       img = img.resize((self.image_shape[0], self.image_shape[1]), Image.ANTIALIAS)
                       gtboxes = np.array(newbox)
                   else:
                       original_size = img.size
                       self.image_shape=(self.image_shape[1],self.image_shape[0])
                       x_scale = self.image_shape[0] / original_size[0]
                       y_scale = self.image_shape[1] / original_size[1]
                       newbox = []
                       for i in range(len(gtboxes)):
                           newbox.append(
                               [gtboxes[i][0] * x_scale, gtboxes[i][1] * y_scale, gtboxes[i][2] * x_scale,
                                gtboxes[i][3] * y_scale]
                           )
                       img = img.resize((self.image_shape[0], self.image_shape[1]), Image.ANTIALIAS)
                       gtboxes = np.array(newbox)

                if self.base_model_name=='vgg16':
                    scale=16
                else:
                    scale=32
                w, h = img.size
                print('wh{}.{}'.format(w,h))
                if turn==False:
                    [cls, regr], _ = cal_rpn(imgsize=(w, h), featuresize=(int(h / scale), int(w / scale)),
                                             scale=scale, gtboxes=gtboxes,
                                             iou_positive=self.iou_positive, iou_negative=self.iou_negative,
                                             rpn_total_num=self.rpn_total_num, rpn_positive_num=self.rpn_positive_num
                                             )

                    img = np.array(img)
                    img = (img / 255.0) * 2.0 - 1.0
                else:
                    [cls,regr],_=cp.cal_rpn((h,w), (int(h / scale), int(w / scale)), scale,gtboxes)
                    img=np.array(img)
                    img=img-cp.IMAGE_MEAN
                regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])
                cls = np.expand_dims(cls, axis=0)
                input_batch_imgs.append(img)
                batch_cls.append(cls)
                batch_regr.append(regr)
                batch_count+=1
                if batch_count>=self.batch_size:
                    input_imgs=np.array(input_batch_imgs)
                    np_batch_cls=np.array(batch_cls)
                    np_batch_regr=np.array(batch_regr)

                    yield input_imgs,{'rpn_class_finnal':np_batch_cls,'rpn_regr_finnal':np_batch_regr}

                    input_batch_imgs = []
                    batch_cls = []
                    batch_regr = []
                    batch_count = 0
            np.random.shuffle(self.input_index)
    def generator_V2(self):
        input_batch_imgs=[]
        batch_cls=[]
        batch_regr=[]
        batch_count = 0
        all_image_name=os.listdir(self.image_dir)
        while True:

            for i in self.input_index:
                img_name=all_image_name[i]
                label_name=img_name.replace('.jpg','.xml')
                img=Image.open(os.path.join(self.image_dir,img_name))
                gtboxes, xml_filename = readxml(os.path.join(self.label_dir,label_name))
                if xml_filename!=img_name:
                    raise Exception('read xml error')
                ow, oh = img.size
                # print('ratate前 {} {}'.format(ow,oh))
                img = rotate(os.path.join(self.image_dir, img_name))
                # print('ratate 后 {} {}'.format(img.size[0], img.size[1]))
                newbox = []
                for i in gtboxes:
                    newbox.append([i[1], ow - i[2], i[3], ow - i[0]])
                gtboxes = np.array(newbox)

                if self.image_shape!=None:
                       original_size = img.size
                       x_scale = self.image_shape[1] / original_size[0]
                       y_scale = self.image_shape[0] / original_size[1]
                       newbox = []
                       for i in range(len(gtboxes)):
                           newbox.append(
                               [gtboxes[i][0] * x_scale, gtboxes[i][1] * y_scale, gtboxes[i][2] * x_scale,
                                gtboxes[i][3] * y_scale]
                           )
                       img = img.resize((self.image_shape[1], self.image_shape[0]), Image.ANTIALIAS)
                       gtboxes = np.array(newbox)

                if self.base_model_name=='vgg16':
                    scale=16
                else:
                    scale=32
                w, h = img.size
                # print('wh{}.{}'.format(w,h))
                [cls, regr], _ = cp.cal_rpn((h, w), (int(h / scale), int(w / scale)), scale, gtboxes)
                img = np.array(img)
                img = img - cp.IMAGE_MEAN
                regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])
                cls = np.expand_dims(cls, axis=0)
                input_batch_imgs.append(img)
                batch_cls.append(cls)
                batch_regr.append(regr)
                batch_count+=1
                if batch_count>=self.batch_size:
                    input_imgs=np.array(input_batch_imgs)
                    np_batch_cls=np.array(batch_cls)
                    np_batch_regr=np.array(batch_regr)

                    yield input_imgs,{'rpn_class_finnal':np_batch_cls,'rpn_regr_finnal':np_batch_regr}

                    input_batch_imgs = []
                    batch_cls = []
                    batch_regr = []
                    batch_count = 0
            np.random.shuffle(self.input_index)


class ctpn_tf_model(object):
    def __init__(self,img_dir,label_dir,version,optimizer,lr,ck_dir,log_dir,image_shape=None,k=10,val_size=2000, base_model_name='vgg16',epoch=100,initial_epoch=0,
                 iou_positive=0.7, iou_negative=0.3, iou_select=0.7, rpn_positive_num=150, rpn_total_num=300,train_bs=50,val_bs=50,trained_weight=None):
        """
        initial arguments
        :param img_dir:
        :param label_dir:
        :param version:
        :param optimizer:
        :param lr:
        :param ck_dir:
        :param log_dir:
        :param image_shape:
        :param k:
        :param val_size:
        :param base_model_name:
        :param epoch:
        :param initial_epoch:
        :param iou_positive:
        :param iou_negative:
        :param iou_select:
        :param rpn_positive_num:
        :param rpn_total_num:
        :param train_bs:
        :param val_bs:
        :param trained_weight: load the trained weight for the next training
        """

        self.img_dir=img_dir
        self.label_dir=label_dir
        if os.path.isdir(self.img_dir)==False or os.path.isdir(self.label_dir)==False:
            raise Exception('folder error')
        if len(os.listdir(self.img_dir))!=len(os.listdir(self.label_dir)):
            raise Exception('folder file length error!')
        self.version=version
        self.img_shape=image_shape#(w,h,3)
        self.k=k
        self.val_size=val_size
        self.base_model_name=base_model_name
        self.iou_positive=iou_positive
        self.iou_negative=iou_negative
        self.iou_select=iou_select
        self.rpn_positive_num=rpn_positive_num
        self.rpn_total_num=300
        self.rpn_total_num=rpn_total_num
        self.train_bs=train_bs
        self.val_bs=val_bs
        self.epoch=epoch
        self.optimizer=optimizer
        self.initial_epoch=initial_epoch
        self.trained_weight=trained_weight
        self.lr=lr
        self.ck_dir=ck_dir
        self.log_dir=log_dir
        all_index = np.array(range(len(os.listdir(self.img_dir))))
        np.random.shuffle(all_index)
        self.val_index=np.random.choice(all_index,size=self.val_size,replace=False)
        self.train_index=[i for i in all_index if i not in self.val_index]

    def train(self):
        """
        train function: first, generate the data gen, then fit_generator
        :return:
        """
        train_gen=DataGenerator(image_dir=self.img_dir,label_dir=self.label_dir,mode='train',batch_size=self.train_bs,
                                image_shape=self.img_shape,k=self.k,iou_select=self.iou_select,input_index=self.train_index,
                                iou_negative=self.iou_negative,iou_positive=self.iou_positive,
                                base_model_name=self.base_model_name,rpn_total_num=self.rpn_total_num,
                                rpn_positive_num=self.rpn_positive_num)
        val_gen=DataGenerator(image_dir=self.img_dir,label_dir=self.label_dir,mode='val',image_shape=self.img_shape,
                               k=self.k,iou_select=self.iou_select,batch_size=self.train_bs,input_index=self.val_index,
                                iou_negative=self.iou_negative,iou_positive=self.iou_positive,
                                base_model_name=self.base_model_name,rpn_total_num=self.rpn_total_num,
                                rpn_positive_num=self.rpn_positive_num)
        shape=(None,None,3) if self.img_shape==None else (self.img_shape[0],self.img_shape[1],self.img_shape[2])
        self.train_model,self.predict_model=proposal_model(base_model_name=self.base_model_name,k=self.k,
                                                           trained_weight=self.trained_weight,
                                                           image_shape=shape,
                                                           bs=self.train_bs
                                                           ).model_ctpn()
        if self.optimizer.lower()=='adam':
            opt=optimizers.Adam(self.lr)
        elif self.optimizer.lower()=='sgd':
            opt=optimizers.SGD(self.lr,decay=1e-6, momentum=0.9, nesterov=True)
        elif self.optimizer.lower()=='ada':
            opt=optimizers.Adadelta(lr=self.lr)
        else:
            opt=optimizers.SGD(self.lr,decay=1e-6, momentum=0.9, nesterov=True)
        LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='min', verbose=1)
        t=os.path.join(self.ck_dir,self.version)
        if os.path.isdir(t)==False:
            os.mkdir(t)
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(t, 'ctpn-keras--{epoch:02d}--{val_loss:.5f}.hdf5'),
            monitor='val_loss', save_best_only=False,
            verbose=1, mode='min', period=1)
        TB = TensorBoard(log_dir=self.log_dir)
        self.train_model.compile(
            optimizer=opt,
            loss={'rpn_class_finnal':rpn_cls_loss,'rpn_regr_finnal':rpn_regr_loss},
            loss_weights={'rpn_class_finnal':1.0,'rpn_regr_finnal':1.0}
        )
        if self.trained_weight!=None:
            self.train_model.load_weights(self.trained_weight)
        his=self.train_model.fit_generator(
            generator=train_gen.generator_V2(),
            steps_per_epoch=train_gen.steps_per_epoch,
            validation_data=val_gen.generator_V2(),
            validation_steps=val_gen.steps_per_epoch,
            epochs=self.epoch,
            initial_epoch=self.initial_epoch,
            callbacks=[checkpoint, early_stop, LR, TB]
        )
        print(his)

    def predict(self,dir,predict_model_path,resize=False,turn=True,mode=1):
        """

        :param dir:a dir or a image path
        :param predict_model_path:
        :param mode:1:show image,  2 return predict result
        :return:
        """
        shape = (None, None, 3) if self.img_shape == None else (self.img_shape[0], self.img_shape[1], self.img_shape[2])
        train_model, predict_model = proposal_model(base_model_name=self.base_model_name, k=self.k,
                                                              trained_weight=self.trained_weight,
                                                              image_shape=shape,
                                                              bs=1
                                                              ).model_ctpn()
        predict_model.load_weights(predict_model_path)
        if self.base_model_name=='vgg16':
            scale=16
        else:
            scale=32
        if os.path.isdir(dir):
            img_list=[os.path.join(dir,i) for i in os.listdir(dir)]
        else:
            img_list=[dir]

        original_shape_list=[]
        radios=[]
        for img_path in img_list:
            img=Image.open(img_path)
            origin_w,origin_h=img.size
            if turn:
                img=rotate(img_path)
            w,h=origin_w,origin_h
            if resize:
                img=img.resize((self.img_shape[1], self.img_shape[0]), Image.ANTIALIAS)
                x_radio,y_radio=self.img_shape[1]/img.size[0],self.img_shape[0]/img.size[1]
                radios.append([x_radio,y_radio])
                w,h=self.img_shape[1],self.img_shape[0]
            else:
                if origin_w<scale and origin_h<scale:
                    img=img.resize((scale,scale),Image.ANTIALIAS)
                    origin_w,origin_h=(scale,scale)
                    w,h=origin_w,origin_h
            original_shape_list.append([origin_w, origin_h])
            img_pil=img
            img = np.array(img)
            if turn:
                img = img - cp.IMAGE_MEAN
            else:
                img=(img / 255.0) * 2.0 - 1.0
            img=np.expand_dims(img,axis=0)
            cls,regr,cls_prob=predict_model.predict(img)

            anchors = cp.gen_anchor((h // scale, w // scale), scale)

            bbox=cp.bbox_transfor_inv(anchors,regr)

            fg=np.where(cls_prob[0,:,1]>self.iou_select)[0]

            select_anchor=bbox[fg,:]
            select_score=cls_prob[0,fg,1]
            select_anchor=select_anchor.astype('int32')
            #filter box
            keep_index=cp.filter_bbox(select_anchor,scale)

            select_anchor=select_anchor[keep_index]
            select_score=select_score[keep_index]

            select_score=np.reshape(select_score,(select_score.shape[0],1))
            nmsbox=np.hstack((select_anchor,select_score))
            keep=cp.nms(nmsbox,1-self.iou_select)
            select_anchor=select_anchor[keep]
            select_score=select_score[keep]
            textConn = cp.TextProposalConnectorOriented()
            text = textConn.get_text_lines(select_anchor, select_score, [h, w])
            text=text.astype(int)
            drawRect(select_anchor,img_pil)
            break

if __name__=='__main__':
   m=ctpn_tf_model(
       img_dir='D:\py_projects\data_new\data_new\data\\train_img',
       label_dir='D:\py_projects\data_new\data_new\data\\annotation',
       version='keras_ctpn_v3_turn',
       optimizer='sgd',
       log_dir='D:\py_projects\VTD\logs',
       ck_dir='D:\py_projects\VTD\model',
       image_shape=(256,512,3),
       initial_epoch=0,
       train_bs=5,
       val_bs=5,
       lr=0.001
   )
   m.train()
   # m = ctpn_tf_model(
   #     img_dir='D:\python_projects\data_new\data\\train_img',
   #     label_dir='D:\python_projects\data_new\data\\annotation',
   #     version='keras_ctpn_sgd_top_ramdon',
   #     optimizer='sgd',
   #     log_dir='D:\\vertical_text_detection\logs',
   #     ck_dir='D:\\vertical_text_detection\model',
   #     image_shape=(256, 512, 3),
   #     initial_epoch=0,
   #     train_bs=20,
   #     val_bs=20,
   #     lr=0.001,
   #     # trained_weight='D:\\vertical_text_detection\model\keras_ctpn_adam\ctpn-keras--05--0.47603.hdf5'
   # )
   # m.train()