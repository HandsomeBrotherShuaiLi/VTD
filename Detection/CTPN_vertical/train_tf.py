import numpy as np
import os
from PIL import Image
from Detection.CTPN_vertical.utils import cal_rpn,readxml
from Detection.CTPN_vertical.model_keras import proposal_model
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping,TensorBoard
def rpn_regr_loss(y_true,y_pred):
    print('rpn_regr_loss')
    print(y_true.shape)
    print(y_pred.shape)
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
    print('rpn_cls_loss')
    print(y_true.shape)
    print(y_pred.shape)
    y_true = y_true[0][0]
    cls_keep = tf.where(tf.not_equal(y_true, -1))[:, 0]
    cls_true = tf.gather(y_true, cls_keep)
    cls_pred = tf.gather(y_pred[0], cls_keep)
    cls_true = tf.cast(cls_true, 'int64')
    # loss = K.sparse_categorical_crossentropy(cls_true,cls_pred,from_logits=True)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cls_true, logits=cls_pred)
    return K.switch(tf.size(loss) > 0, K.clip(K.mean(loss), 0, 10), K.constant(0.0))


class DataGenerator(object):
    def __init__(self,image_dir, mode,label_dir,val_size=2000,image_shape=None,k=10, base_model_name='vgg16',
                 iou_positive=0.7, iou_negative=0.3, iou_select=0.7, rpn_positive_num=150, rpn_total_num=300,train_bs=50,val_bs=50
                 ):
        self.mode=mode
        assert self.mode in ['train','val']
        self.image_dir=image_dir
        self.label_dir=label_dir
        if os.path.isdir(self.image_dir)==False or os.path.isdir(self.label_dir)==False:
            raise Exception('invalid directory,please check your image dir and label dir')
        if len(os.listdir(self.image_dir))!=len(os.listdir(self.label_dir)):
            raise Exception('Different lengths of input dir file,in fact, they are supposed to have the same file number')
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
        if self.image_shape==None:#else (w,h)
            self.train_bs,self.val_bs=1,1
        self.total_len = len(os.listdir(self.image_dir))
        self.all_index = np.array(range(self.total_len))
        if self.mode=='train':
            self.steps_per_epoch=((self.total_len)-self.val_size)//self.train_bs
        else:
            self.steps_per_epoch=self.val_size//self.val_bs
    def __shuffle__(self):
        np.random.shuffle(self.all_index)
        self.val_index=np.random.choice(self.all_index,size=self.val_size,replace=False)
        self.train_index=[i for i in self.all_index if i not in self.val_index]
    def generator(self):
        input_batch_imgs=[]
        batch_cls=[]
        batch_regr=[]
        batch_count = 0
        all_image_name=os.listdir(self.image_dir)
        while True:
            self.__shuffle__()
            mapping = {'train': self.train_index, 'val': self.val_index}
            img_index = mapping[self.mode]
            batch_size_mapping = {'train': self.train_bs, 'val': self.val_bs}
            for i in img_index:
                img_name=all_image_name[i]
                label_name=img_name.replace('.jpg','.xml')
                img=Image.open(os.path.join(self.image_dir,img_name))
                gtboxes, xml_filename = readxml(os.path.join(self.label_dir,label_name))
                if xml_filename!=img_name:
                    raise Exception('read xml error')
                if self.image_shape!=None:
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
                w,h=img.size

                if self.base_model_name=='vgg16':
                    scale=16
                else:
                    scale=32
                [cls,regr],_=cal_rpn(imgsize=(w,h),featuresize=(int(h/scale),int(w/scale)),
                                     scale=scale,gtboxes=gtboxes,
                                     iou_positive=self.iou_positive, iou_negative=self.iou_negative,
                                     rpn_total_num=self.rpn_total_num, rpn_positive_num=self.rpn_positive_num
                                     )

                img = np.array(img)
                img = (img / 255.0) * 2.0 - 1.0
                regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])
                cls = np.expand_dims(cls, axis=0)
                input_batch_imgs.append(img)
                batch_cls.append(cls)
                batch_regr.append(regr)
                batch_count+=1
                if batch_count>=batch_size_mapping[self.mode]:
                    input_imgs=np.array(input_batch_imgs)
                    np_batch_cls=np.array(batch_cls)
                    np_batch_regr=np.array(batch_regr)
                    # print(input_imgs.shape,np_batch_regr.shape,np_batch_cls.shape)
                    yield input_imgs,{'rpn_class_finnal':np_batch_cls,'rpn_regr_finnal':np_batch_regr}

                    input_batch_imgs = []
                    batch_cls = []
                    batch_regr = []
                    batch_count = 0

class ctpn_tf_model_trainer(object):
    def __init__(self,img_dir,label_dir,version,optimizer,lr,ck_dir,log_dir,image_shape=None,k=10,val_size=2000, base_model_name='vgg16',epoch=100,initial_epoch=0,
                 iou_positive=0.7, iou_negative=0.3, iou_select=0.7, rpn_positive_num=150, rpn_total_num=300,train_bs=50,val_bs=50,trained_weight=None):
        self.img_dir=img_dir
        self.label_dir=label_dir
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

    def train(self):
        train_gen=DataGenerator(image_dir=self.img_dir,label_dir=self.label_dir,mode='train',val_size=self.val_size,
                                val_bs=self.val_bs,image_shape=self.img_shape,k=self.k,iou_select=self.iou_select,
                                iou_negative=self.iou_negative,iou_positive=self.iou_positive,train_bs=self.train_bs,
                                base_model_name=self.base_model_name,rpn_total_num=self.rpn_total_num,
                                rpn_positive_num=self.rpn_positive_num)
        val_gen=DataGenerator(image_dir=self.img_dir,label_dir=self.label_dir,mode='val',val_size=self.val_size,
                                val_bs=self.val_bs,image_shape=self.img_shape,k=self.k,iou_select=self.iou_select,
                                iou_negative=self.iou_negative,iou_positive=self.iou_positive,train_bs=self.train_bs,
                                base_model_name=self.base_model_name,rpn_total_num=self.rpn_total_num,
                                rpn_positive_num=self.rpn_positive_num)
        shape=(None,None,3) if self.img_shape==None else (self.img_shape[1],self.img_shape[0],self.img_shape[2])
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
        LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=11, mode='min', verbose=1)
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
            generator=train_gen.generator(),
            steps_per_epoch=train_gen.steps_per_epoch,
            validation_data=val_gen.generator(),
            validation_steps=val_gen.steps_per_epoch,
            epochs=self.epoch,
            initial_epoch=self.initial_epoch,
            callbacks=[checkpoint, early_stop, LR, TB]
        )
        print(his)
if __name__=='__main__':
   m=ctpn_tf_model_trainer(
       img_dir='D:\py_projects\data_new\data_new\data\\train_img',
       label_dir='D:\py_projects\data_new\data_new\data\\annotation',
       version='keras_ctpn_v1',
       optimizer='sgd',
       log_dir='D:\py_projects\VTD\logs',
       ck_dir='D:\py_projects\VTD\model',
       image_shape=(256,512,3),
       initial_epoch=0,
       train_bs=50,
       val_bs=50,
       lr=0.001
   )
   m.train()














