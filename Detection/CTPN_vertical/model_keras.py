from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16,InceptionResNetV2,DenseNet121,ResNet50,Xception
from tensorflow.keras import Model,Input
import tensorflow.keras.backend as K
import tensorflow as tf
def backend_reshape(x):
    t=x.get_shape()
    return K.reshape(x,(t[0]*t[2],t[1],t[-1]))
def backend_reshape_v2(x):
    t = x.get_shape()
    return K.reshape(x, (t[0] * t[1], t[2], t[-1]))

def backend_reshape_2(x):
    x1,x2=x
    t=x2.get_shape()
    return K.reshape(x1,(t[0],t[1],t[2],256))

def backend_reshape_3(x):
    x1,x2=x
    k=K.cast(x2,dtype=tf.int32)
    t=x1.get_shape()
    return K.reshape(x1,(t[0],t[1]*t[2]*k,2))


class proposal_model(object):
    def __init__(self,bs,base_model_name=None,image_shape=(None,None,3),k=10,trained_weight=None):
        self.base_model_name=base_model_name
        self.k=k
        self.trained_weight=trained_weight
        self.image_shape=image_shape
        self.bs=bs

    def model_ctpn(self):
        input_layer=Input(shape=self.image_shape,batch_size=self.bs,name='input_1')
        if self.base_model_name.lower()=='vgg16' or self.base_model_name==None:
            self.scale=16
            base_model=VGG16(include_top=False,input_tensor=input_layer)
            base_model.trainable=True
            # input_layer=base_model.input
            base_output_layer=base_model.get_layer('block5_conv3').output
        elif self.base_model_name.lower()=='resnet50':
            base_model=ResNet50(include_top=False,input_tensor=input_layer)
            self.scale=32
            base_model.trainable=True
            # input_layer=base_model.input
            base_output_layer=base_model.output
        elif self.base_model_name.lower()=='inception':
            base_model=InceptionResNetV2(include_top=False,input_tensor=input_layer)
            base_model.trainable=True
            # input_layer=base_model.input
            base_output_layer=base_model.output
            self.scale=32
        elif self.base_model_name.lower()=='xception':
            base_model=Xception(include_top=False,input_tensor=input_layer)
            base_model.trainable = True
            # input_layer = base_model.input
            base_output_layer = base_model.output
            self.scale=32
        else:
            base_model=DenseNet121(include_top=False,input_tensor=input_layer)
            base_model.trainable = True
            # input_layer = base_model.input
            base_output_layer = base_model.get_layer('conv5_block16_concat').output
            self.scale = 32
        layer_base=layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',name='share_layer',
                            kernel_initializer='he_normal')(base_output_layer)
        #这里得到share layer之后,把输出层(bs,h,w,c)->(bs*w,h,c),因为我们是垂直列识别，把每一个width的列提出进行训练
        shape=layer_base.get_shape()

        layer=layers.Lambda(backend_reshape,output_shape=(shape[1],shape[-1]),name='lambda_reshape_1')(layer_base)


        layer=layers.Bidirectional(layers.LSTM(128,return_sequences=True,kernel_initializer='he_normal'),merge_mode='concat')(layer)


        layer=layers.Lambda(backend_reshape_2,output_shape=(shape[1],shape[2],256),name='lambda_reshape_2')([layer,layer_base])
        layer=layers.Conv2D(128,(1,1),padding='same',activation='relu',kernel_initializer='he_normal',name='lstm_fc')(layer)

        cls=layers.Conv2D(self.k*2,(1,1),padding='same',kernel_initializer='he_normal',activation='linear',name='rpn_class')(layer)
        regr=layers.Conv2D(self.k*2,(1,1),padding='same',kernel_initializer='he_normal',activation='linear',name='rpn_regr')(layer)
        shape2=cls.get_shape()
        # #bs,H*W*k,2
        cls=layers.Reshape(target_shape=(shape2[1]*shape2[2]*self.k,2),batch_size=shape2[0],name='rpn_class_finnal')(cls)
        cls_prob=layers.Activation('softmax',name='rpn_cls_softmax')(cls)
        shape3=regr.get_shape()
        regr=layers.Reshape(target_shape=(shape3[1]*shape3[2]*self.k,2),batch_size=shape3[0],name='rpn_regr_finnal')(regr)

        predict_model=Model(input_layer,[cls,regr,cls_prob])
        train_model=Model(input_layer,[cls,regr])
        train_model.summary()
        return train_model,predict_model

    def model_ctpn_v2(self):
        input_layer = Input(shape=self.image_shape, batch_size=self.bs, name='input_1')
        if self.base_model_name.lower() == 'vgg16' or self.base_model_name == None:
            self.scale = 16
            base_model = VGG16(include_top=False, input_tensor=input_layer)
            base_model.trainable = True
            # input_layer=base_model.input
            base_output_layer = base_model.get_layer('block5_conv3').output
        elif self.base_model_name.lower() == 'resnet50':
            base_model = ResNet50(include_top=False, input_tensor=input_layer)
            self.scale = 32
            base_model.trainable = True
            # input_layer=base_model.input
            base_output_layer = base_model.output
        elif self.base_model_name.lower() == 'inception':
            base_model = InceptionResNetV2(include_top=False, input_tensor=input_layer)
            base_model.trainable = True
            # input_layer=base_model.input
            base_output_layer = base_model.output
            self.scale = 32
        elif self.base_model_name.lower() == 'xception':
            base_model = Xception(include_top=False, input_tensor=input_layer)
            base_model.trainable = True
            # input_layer = base_model.input
            base_output_layer = base_model.output
            self.scale = 32
        else:
            base_model = DenseNet121(include_top=False, input_tensor=input_layer)
            base_model.trainable = True
            # input_layer = base_model.input
            base_output_layer = base_model.get_layer('conv5_block16_concat').output
            self.scale = 32
        layer_base = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='share_layer',
                                   kernel_initializer='he_normal')(base_output_layer)
        # 这里得到share layer之后,把输出层(bs,h,w,c)->(bs*h,w,c),因为我们是垂直列识别，把每一个width的列提出进行训练
        shape = layer_base.get_shape()

        layer = layers.Lambda(backend_reshape_v2, output_shape=(shape[2], shape[-1]), name='lambda_reshape_1')(layer_base)

        layer = layers.Bidirectional(layers.LSTM(128, return_sequences=True, kernel_initializer='he_normal'),
                                     merge_mode='concat')(layer)

        layer = layers.Lambda(backend_reshape_2, output_shape=(shape[1], shape[2], 256), name='lambda_reshape_2')(
            [layer, layer_base])
        layer = layers.Conv2D(128, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal',
                              name='lstm_fc')(layer)

        cls = layers.Conv2D(self.k * 2, (1, 1), padding='same', kernel_initializer='he_normal', activation='linear',
                            name='rpn_class')(layer)
        regr = layers.Conv2D(self.k * 2, (1, 1), padding='same', kernel_initializer='he_normal', activation='linear',
                             name='rpn_regr')(layer)
        shape2 = cls.get_shape()
        # #bs,H*W*k,2
        cls = layers.Reshape(target_shape=(shape2[1] * shape2[2] * self.k, 2), batch_size=shape2[0],
                             name='rpn_class_finnal')(cls)
        cls_prob = layers.Activation('softmax', name='rpn_cls_softmax')(cls)
        shape3 = regr.get_shape()
        regr = layers.Reshape(target_shape=(shape3[1] * shape3[2] * self.k, 2), batch_size=shape3[0],
                              name='rpn_regr_finnal')(regr)

        predict_model = Model(input_layer, [cls, regr, cls_prob])
        train_model = Model(input_layer, [cls, regr])
        train_model.summary()
        return train_model, predict_model

if __name__=='__main__':
    #这里的image shape 是（h,w,3)
    m=proposal_model(base_model_name='vgg16',image_shape=(1024,512,3),bs=2).model_ctpn()


