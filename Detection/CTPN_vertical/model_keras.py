from keras.applications import VGG16,VGG19,DenseNet121,ResNet50
from keras.layers import Dense,Conv2D,BatchNormalization,Lambda,LSTM,GRU,Activation,Bidirectional,Reshape
from keras.models import Model,Input
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.contrib import slim
import keras.backend as K
def reshape(x):
    """
    因为我们是垂直文本，所以把每一列当做新的feature map sample * bs，形成新的bs,h,1,c
    shape(bs,h,w,c)--> (bs*w,h,c)
    :param x:
    :return:
    """
    shape=tf.shape(x)
    return tf.reshape(x,[shape[0]*shape[2],shape[1],shape[-1]])

def reshape_lstm(x):
    net,conv_layer=x
    conv_shape=tf.shape(conv_layer)
    return tf.reshape(net,[conv_shape[0],conv_shape[1],conv_shape[2],256])

def reshape_fc(x):
    b = tf.shape(x)
    x = tf.reshape(x, [b[0], b[1] * b[2] * 35, 2])  # (N, H x W x 10, 2)
    return x

def make_var(name, shape, initializer=None):
    return tf.get_variable(name, shape, initializer=initializer)


def Bilstm(net, input_channel, hidden_unit_num, output_channel, scope_name):
    # width--->time step
    with tf.variable_scope(scope_name) as scope:
        shape = tf.shape(net)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [N * W, H, C])
        net.set_shape([None, None, input_channel])

        lstm_fw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)

        lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, net, dtype=tf.float32)
        lstm_out = tf.concat(lstm_out, axis=-1)

        lstm_out = tf.reshape(lstm_out, [N * H * W, 2 * hidden_unit_num])

        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [2 * hidden_unit_num, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        outputs = tf.matmul(lstm_out, weights) + biases

        outputs = tf.reshape(outputs, [N, H, W, output_channel])
        return outputs


def lstm_fc(net, input_channel, output_channel, scope_name):
    with tf.variable_scope(scope_name) as scope:
        shape = tf.shape(net)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [N * H * W, C])

        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [input_channel, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        output = tf.matmul(net, weights) + biases
        output = tf.reshape(output, [N, H, W, output_channel])
    return output


def _rpn_loss_regr(y_true, y_pred):
    """
    smooth L1 loss
    y_ture [1][HXWX10][3] (class,regr)
    y_pred [1][HXWX10][2] (reger)
    """

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


def _rpn_loss_cls(y_true, y_pred):
    """
    softmax loss
    y_true [1][1][HXWX10] class
    y_pred [1][HXWX10][2] class
    """
    y_true = y_true[0][0]
    cls_keep = tf.where(tf.not_equal(y_true, -1))[:, 0]
    cls_true = tf.gather(y_true, cls_keep)
    cls_pred = tf.gather(y_pred[0], cls_keep)
    cls_true = tf.cast(cls_true, 'int64')
    # loss = K.sparse_categorical_crossentropy(cls_true,cls_pred,from_logits=True)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cls_true, logits=cls_pred)
    return K.switch(tf.size(loss) > 0, K.clip(K.mean(loss), 0, 10), K.constant(0.0))


class proposed_model(object):
    def __init__(self,base_net,lr,image_shape,k=35,trainable=True,weight_path=None):
        assert base_net in ['vgg16','vgg19','densenet121','resnet50']
        self.base_net_name=base_net
        self.lr=lr
        self.k=k
        self.image_shape=image_shape
        self.trainable=trainable
        self.weight_path=weight_path

    def model(self,isGRU=False):
        if self.base_net_name.lower()=='vgg16':
            base_net=VGG16(include_top=False,input_shape=self.image_shape, weights='imagenet')
            base_net.trainable=True if self.trainable else False
            input=base_net.input
            sub_output=base_net.get_layer('block5_conv3').output
        elif self.base_net_name.lower()=='vgg19':
            base_net=VGG19(include_top=False,input_shape=self.image_shape, weights='imagenet')
            base_net.trainable = True if self.trainable else False
            input=base_net.input
            sub_output=base_net.get_layer('block5_conv4').output
        elif self.base_net_name.lower()=='densenet121':
            base_net=DenseNet121(include_top=False,input_shape=self.image_shape, weights='imagenet')
            base_net.trainable = True if self.trainable else False
            input = base_net.input
            sub_output=base_net.get_layer('conv5_block16_concat').output
        else:
            base_net=ResNet50(include_top=False,input_shape=self.image_shape, weights='imagenet')
            base_net.trainable =True if self.trainable else False
            input=base_net.input
            sub_output=base_net.output

        #完成了基础网络结构，开始3x3 feature map 的sliding window
        layer=Conv2D(512,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',name='ctpn_rpn_conv1')(sub_output)
        conv_layer=layer
        layer=Lambda(reshape,name='reshape_layer1',output_shape=(None,512))(layer)
        if isGRU==False:
            layer = Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='he_normal'), merge_mode='concat',
                                  name='bilstm')(layer)
        else:
            layer = Bidirectional(GRU(128, return_sequences=True, kernel_initializer='he_normal'), merge_mode='concat',
                                  name='bilstm')(layer)
        layer=Lambda(reshape_lstm,output_shape=(None,None,256),name='reshape_layer2')([layer,conv_layer])
        layer=Conv2D(512,(1,1),padding='same',activation='relu',name='lstm_fc')(layer)

        cls=Conv2D(self.k*2,(1,1),padding='same',activation='linear',name='rpn_class_original')(layer)
        regr=Conv2D(self.k*2,(1,1),padding='same',activation='linear',name='rpn_regression_original')(layer)

        cls=Lambda(reshape_fc,output_shape=(None,2),name='rpn_class')(cls)
        cls_prob=Activation('softmax',name='rpn_cls_pred')(cls)

        regr=Lambda(reshape_fc,output_shape=(None,2),name='rpn_regression')(regr)

        train_model=Model(input,[cls,regr])
        predict_model=Model(input,[cls,regr,cls_prob])
        adam=Adam(self.lr)
        train_model.compile(
            optimizer=adam,
            loss={
                'rpn_regression':_rpn_loss_regr,
                'rpn_class':_rpn_loss_cls
            },
            loss_weights={
                'rpn_regression':1.0,
                'rpn_class':1.0
            }
        )
        train_model.summary()
        predict_model.summary()
        return train_model, predict_model

if __name__=='__main__':
    m1,m2=proposed_model(
        base_net='densenet121',lr=1e-5,image_shape=(1024,512,3),k=10
    ).model()