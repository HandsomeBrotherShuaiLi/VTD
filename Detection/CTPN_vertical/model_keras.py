import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16,InceptionResNetV2,DenseNet121,ResNet50,Xception
from tensorflow.keras import Model,Input
import tensorflow.keras.optimizers as optimizers
class proposal_model(object):
    def __init__(self,base_model_name=None,image_shape=(None,None,3),k=10,trained_weight=None):
        self.base_model_name=base_model_name
        self.k=k
        self.trained_weight=trained_weight
        self.image_shape=image_shape
    def model(self):
        if self.base_model_name.lower()=='vgg16' or self.base_model_name==None:
            self.scale=16
            base_model=VGG16(include_top=False,input_shape=self.image_shape)
            base_model.trainable=True
            input_layer=base_model.input
            base_output_layer=base_model.get_layer('block5_conv3').output
        elif self.base_model_name.lower()=='resnet50':
            base_model=ResNet50(include_top=False,input_shape=self.image_shape)
            self.scale=32
            base_model.trainable=True
            input_layer=base_model.input
            base_output_layer=base_model.output
        elif self.base_model_name.lower()=='inception':
            base_model=InceptionResNetV2(include_top=False,input_shape=self.image_shape)
            base_model.trainable=True
            input_layer=base_model.input
            base_output_layer=base_model.output
            self.scale=32
        elif self.base_model_name.lower()=='xception':
            base_model=Xception(include_top=False,input_shape=self.image_shape)
            base_model.trainable = True
            input_layer = base_model.input
            base_output_layer = base_model.output
            self.scale=32
        else:
            base_model=DenseNet121(include_top=False,input_shape=self.image_shape)
            base_model.trainable = True
            input_layer = base_model.input
            base_output_layer = base_model.get_layer('conv5_block16_concat').output
            self.scale = 32
        layer=layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',name='share_layer',
                            kernel_initializer='he_normal')(base_output_layer)
        #这里得到share layer之后,把输出层(bs,h,w,c)->(bs*w,h,c),因为我们是垂直列识别，把每一个width的列提出进行训练
        shape=layer.get_shape()
        # print(shape)
        layer=layers.Reshape(target_shape=(shape[1],shape[-1]),batch_size=shape[0]*shape[2])(layer)
        layer=layers.Bidirectional(layers.LSTM(128,return_sequences=True,kernel_initializer='he_normal'),merge_mode='concat')(layer)
        layer=layers.Reshape(target_shape=(shape[1],shape[2],256),batch_size=shape[0])(layer)
        layer=layers.Conv2D(128,(1,1),padding='same',activation='relu',kernel_initializer='he_normal',name='lstm_fc')(layer)

        cls=layers.Conv2D(self.k*2,(1,1),padding='same',kernel_initializer='he_normal',activation='linear',name='rpn_class')(layer)
        regr=layers.Conv2D(self.k*2,(1,1),padding='same',kernel_initializer='he_normal',activation='linear',name='rpn_regr')(layer)
        shape2=cls.get_shape()
        #bs,H*W*k,2
        cls=layers.Reshape(target_shape=(shape2[1]*shape2[2]*self.k,2),batch_size=shape2[0],name='rpn_class_finnal')(cls)
        cls_prob=layers.Activation('softmax',name='rpn_cls_softmax')(cls)
        shape3=regr.get_shape()
        regr=layers.Reshape(target_shape=(shape3[1]*shape3[2]*self.k,2),batch_size=shape3[0],name='rpn_regr_finnal')(regr)
        predict_model=Model(input_layer,[cls,regr,cls_prob])
        train_model=Model(input_layer,[cls,regr])
        return train_model,predict_model
if __name__=='__main__':
    m=proposal_model(base_model_name='vgg16',image_shape=(1024,512,3)).model()


