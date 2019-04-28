# coding=utf-8
from keras import Input, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Concatenate, Conv2D, UpSampling2D, BatchNormalization,Lambda
import keras.backend as K
from Detection.Proposed_Model import cfg

"""
input_shape=(img.height, img.width, 3), height and width must scaled by 32.
So images's height and width need to be pre-processed to the nearest num that
scaled by 32.And the annotations xy need to be scaled by the same ratio 
as height and width respectively.
"""
class ran_block(object):
    def __init__(self,input_tensor):
        self.input_tensor=input_tensor
        self.shape=self.input_tensor.get_shape()
    def reshape_(self,x):
        shape=x.get_shape()
        return K.reshape(x,(int(shape[1])*int(shape[2]),int(shape[-1])))
    def ran_conv(self):
        print(self.shape)
        alpha=Conv2D(int(self.shape[-1])//2,kernel_size=(1,1),padding='same',kernel_initializer='he_normal',name='alpha')(self.input_tensor)
        beta=Conv2D(int(self.shape[-1])//2,kernel_size=(1,1),padding='same',kernel_initializer='he_normal')(self.input_tensor)
        gamma=Conv2D(int(self.shape[-1])//2,kernel_size=(1,1),padding='same',kernel_initializer='he_normal')(self.input_tensor)
        alpha_1=Lambda(self.reshape_,output_shape=(int(alpha.get_shape()[1])*int(alpha.get_shape()[2]),int(alpha.get_shape()[-1])))(alpha)
        beta_1=Lambda(self.reshape_,output_shape=(int(beta.get_shape()[1])*int(beta.get_shape()[2]),int(beta.get_shape()[-1])))(beta)
        

        print(alpha_1.get_shape())


class RAN:
    def __init__(self):
        self.input_img = Input(name='input_img',
                               shape=(512, 512, cfg.num_channels),
                               dtype='float32')
        vgg16 = VGG16(input_tensor=self.input_img,
                      weights='imagenet',
                      include_top=False)
        if cfg.locked_layers:
            # locked first two conv layers
            locked_layers = [vgg16.get_layer('block1_conv1'),
                             vgg16.get_layer('block1_conv2')]
            for layer in locked_layers:
                layer.trainable = False
        self.f = [vgg16.get_layer('block%d_pool' % i).output
                  for i in cfg.feature_layers_range]
        self.f.insert(0, None)
        self.diff = cfg.feature_layers_range[0] - cfg.feature_layers_num

    def g(self, i):
        # i+diff in cfg.feature_layers_range
        assert i + self.diff in cfg.feature_layers_range, \
            ('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(cfg.feature_layers_range)
        if i == cfg.feature_layers_num:
            bn = BatchNormalization()(self.h(i))
            return Conv2D(32, 3, activation='relu', padding='same')(bn)
        else:
            return UpSampling2D((2, 2))(self.h(i))

    def h(self, i):
        # i+diff in cfg.feature_layers_range
        assert i + self.diff in cfg.feature_layers_range, \
            ('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(cfg.feature_layers_range)
        if i == 1:
            return self.f[i]
        else:
            concat = Concatenate(axis=-1)([self.g(i - 1), self.f[i]])
            bn1 = BatchNormalization()(concat)
            conv_1 = Conv2D(128 // 2 ** (i - 2), 1,
                            activation='relu', padding='same')(bn1)
            bn2 = BatchNormalization(name='h_bn_{}'.format(i))(conv_1)
            conv_3 = Conv2D(128 // 2 ** (i - 2), 3,
                            activation='relu', padding='same')(bn2)
            return conv_3

    def ran_network(self):
        before_output = self.g(cfg.feature_layers_num)
        a=ran_block(before_output).ran_conv()
        inside_score = Conv2D(1, 1, padding='same', name='inside_score'
                              )(before_output)
        side_v_code = Conv2D(2, 1, padding='same', name='side_vertex_code'
                             )(before_output)
        side_v_coord = Conv2D(4, 1, padding='same', name='side_vertex_coord'
                              )(before_output)
        east_detect = Concatenate(axis=-1,
                                  name='east_detect')([inside_score,
                                                       side_v_code,
                                                       side_v_coord])
        return Model(inputs=self.input_img, outputs=east_detect)

if __name__ == '__main__':
    ran=RAN()
    model=ran.ran_network()
    model.summary()