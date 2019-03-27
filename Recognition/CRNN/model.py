import keras.backend as K
from keras.layers import Conv2D,MaxPooling2D,Input,Dense,Activation,Reshape,Lambda
from keras.layers import BatchNormalization,Bidirectional
from keras.layers.merge import add,concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM,GRU

class CRNN(object):
    def __init__(self,num_classes,img_w=64,img_h=256,img_c=3,maxLen=40,sp_pool_size=(1,2)):
        """
        :param chars:
        :param num_classes:
        :param img_w:for the training, all images will be resized to w=64,height=256,
               for the test, only to fixed width=64
        :param img_h:
        :param batch_size:
        :param val_batch_size:
        :param downsampling_factor:
        :param maxLen:
        """
        self.num_classes=num_classes
        self.img_w=img_w
        self.img_h=img_h
        self.img_c=img_c
        self.max_text_len=maxLen
        self.sp_pool_size=sp_pool_size
    def ctc_func(self,args):
        y_pred,labels,input_length,label_length=args
        return K.ctc_batch_cost(y_true=labels,y_pred=y_pred,input_length=input_length,label_length=label_length)

    def crnn_v1(self,istraining=True,isGRU=False):
        """
        固定输入尺寸是高：256，宽 64，训练与预测也必须改成这个尺寸
        :param istraining: check whether it is training mode
        :return:
        """
        K.set_image_data_format('channels_last')
        input_shape=(self.img_h,self.img_w,self.img_c)
        #64,256,3
        inputs=Input(name='the_input',
                     shape=input_shape,
                     dtype='float32')
        # None,64,256,3
        layer=Conv2D(64,(3,3),padding='same',name='conv1',
                      kernel_initializer='he_normal')(inputs)# None,64,256,64
        layer=BatchNormalization()(layer)
        layer=Activation('relu')(layer)
        layer=MaxPooling2D(pool_size=(2,2),name='max1')(layer)# None,32,128,64

        layer=Conv2D(128,(3,3),padding='same',name='conv2',
                     kernel_initializer='he_normal')(layer)# None,32,128,128
        layer=BatchNormalization()(layer)
        layer=Activation('relu')(layer)
        layer=MaxPooling2D(pool_size=(2,2),name='max2')(layer)# None,16,64,128

        layer=Conv2D(256,(3,3),padding='same',name='conv3',
                     kernel_initializer='he_normal')(layer)# None,16,64,256
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer=Conv2D(256,(3,3),padding='same',name='conv4',
                     kernel_initializer='he_normal')(layer)# None,16,64,256
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        #Attention!!!!!!!!
        layer=MaxPooling2D(pool_size=self.sp_pool_size,name='max3')(layer)#None,8,64,256

        layer=Conv2D(512,(3,3),padding='same',name='conv5',
                     kernel_initializer='he_normal')(layer)# None,8,64,512
        layer=BatchNormalization()(layer)
        layer=Activation('relu')(layer)
        layer=Conv2D(512,(3,3),padding='same',name='conv6',
                     kernel_initializer='he_normal')(layer)# None,8,64.512
        layer=BatchNormalization()(layer)
        layer=Activation('relu')(layer)
        layer=MaxPooling2D(pool_size=self.sp_pool_size,name='max4')(layer)#None,4,64.512

        layer=Conv2D(512,(2,2),padding='same',name='conv7',
                     kernel_initializer='he_normal')(layer)#None 4,64,512
        layer=BatchNormalization()(layer)
        layer=Activation('relu')(layer)

        #CNN RNN
        #由于我们固定了高度，使得输入一直是256，下采样之后，变成高度是64，我们根据每一行来建立sequeces，所以是feature map的高度
        layer=Reshape(target_shape=(self.img_h//4,-1),name='reshape_layer')(layer)
        #None,64,128
        layer=Dense(128,activation='relu',kernel_initializer='he_normal',
                    name='dense1')(layer)

        if isGRU==False:
            # None,64,256
            # lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal',
            #               name='lstm1')(layer)
            # lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal',
            #               name='lstm2', go_backwards=True)(layer)
            # None,64,256
            # layer = add([lstm_1, lstm_2], name='add_layer')
            layer=Bidirectional(LSTM(256,return_sequences=True,kernel_initializer='he_normal'),merge_mode='sum')(layer)
            layer = BatchNormalization()(layer)
            # lstm_3 = LSTM(256, return_sequences=True, kernel_initializer='he_normal',
            #               name='lstm_3')(layer)
            # lstm_4 = LSTM(256, return_sequences=True, kernel_initializer='he_normal',
            #               name='lstm_4', go_backwards=True)(layer)
            # # None,64,512
            # layer = concatenate([lstm_3, lstm_4], name='concat_layer')
            layer=Bidirectional(LSTM(256,return_sequences=True,kernel_initializer='he_normal'),merge_mode='concat')(layer)
            layer = BatchNormalization()(layer)
        else:
            # gru_1=GRU(256, return_sequences=True, kernel_initializer='he_normal',
            #           name='gru_1')(layer)
            # gru_2=GRU(256, return_sequences=True, kernel_initializer='he_normal',
            #           name='gru_2',go_backwards=True)(layer)
            # layer=add([gru_1,gru_2],name='gru_add_layer')
            layer=Bidirectional(GRU(256,return_sequences=True,kernel_initializer='he_normal'),merge_mode='sum')(layer)
            layer=BatchNormalization()(layer)
            # gru_3=GRU(256, return_sequences=True, kernel_initializer='he_normal',
            #           name='gru_3')(layer)
            # gru_4=GRU(256, return_sequences=True, kernel_initializer='he_normal',
            #           name='gru_4',go_backwards=True)(layer)
            # layer=concatenate([gru_3,gru_4],name='gru_concatenate_layer')
            layer=Bidirectional(GRU(256,return_sequences=True,kernel_initializer='he_normal'),merge_mode='concat')(layer)
            layer=BatchNormalization()(layer)

        layer=Dense(self.num_classes,kernel_initializer='he_normal',
                    name='dense_2')(layer)
        y_pred=Activation(activation='softmax')(layer)
        labels=Input(name='the_labels',shape=[self.max_text_len],dtype='float32')
        input_length=Input(name='input_length',shape=[1],dtype='int64')
        label_length=Input(name='label_length',shape=[1],dtype='int64')
        ctc_loss=Lambda(self.ctc_func,output_shape=(1,),name='ctc_loss')(
            [y_pred,labels,input_length,label_length]
        )
        if istraining:
            return Model(inputs=[inputs,labels,input_length,label_length],outputs=ctc_loss)
        else:
            return Model(inputs=[inputs],outputs=y_pred)
    def crnn_v2(self,istraining=True,isGRU=False):
        """
        不按照Keras的app写了，按照原来的论文。
        :param istraining: check whether it is training mode
        :return:
        """
        K.set_image_data_format('channels_last')
        input_shape=(None,self.img_w,self.img_c)
        #64,256,3
        inputs=Input(name='the_input',
                     shape=input_shape,
                     dtype='float32')
        # None,64,256,3
        layer=Conv2D(64,(3,3),padding='same',name='conv1',
                      kernel_initializer='he_normal')(inputs)# None,64,256,64
        layer=BatchNormalization()(layer)
        layer=Activation('relu')(layer)
        layer=MaxPooling2D(pool_size=(2,2),name='max1')(layer)# None,32,128,64

        layer=Conv2D(128,(3,3),padding='same',name='conv2',
                     kernel_initializer='he_normal')(layer)# None,32,128,128
        layer=BatchNormalization()(layer)
        layer=Activation('relu')(layer)
        layer=MaxPooling2D(pool_size=(2,2),name='max2')(layer)# None,16,64,128

        layer=Conv2D(256,(3,3),padding='same',name='conv3',
                     kernel_initializer='he_normal')(layer)# None,16,64,256
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer=Conv2D(256,(3,3),padding='same',name='conv4',
                     kernel_initializer='he_normal')(layer)# None,16,64,256
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        #Attention!!!!!!!!
        layer=MaxPooling2D(pool_size=self.sp_pool_size,name='max3')(layer)#None,8,64,256

        layer=Conv2D(512,(3,3),padding='same',name='conv5',
                     kernel_initializer='he_normal')(layer)# None,8,64,512
        layer=BatchNormalization()(layer)
        layer=Activation('relu')(layer)
        layer=Conv2D(512,(3,3),padding='same',name='conv6',
                     kernel_initializer='he_normal')(layer)# None,8,64.512
        layer=BatchNormalization()(layer)
        layer=Activation('relu')(layer)
        layer=MaxPooling2D(pool_size=self.sp_pool_size,name='max4')(layer)#None,4,64.512

        layer=Conv2D(512,(2,2),padding='same',name='conv7',
                     kernel_initializer='he_normal')(layer)#None 4,64,512
        layer=BatchNormalization()(layer)
        layer=Activation('relu')(layer)

        #CNN RNN
        #我们固定的是width,height 来提出特征，width固定输入是64, 经过下采样后变成64//16=4，是一个常数
        #而高度变成了 height/4,随输入变化而变化
        layer=Reshape(target_shape=(-1,int(self.img_w//16*512)),name='reshape_layer')(layer)
        #None,None,512
        layer=Dense(512,activation='relu',kernel_initializer='he_normal',
                    name='dense1')(layer)

        if isGRU == False:
            # None,64,256
            # lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal',
            #               name='lstm1')(layer)
            # lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal',
            #               name='lstm2', go_backwards=True)(layer)
            # None,64,256
            # layer = add([lstm_1, lstm_2], name='add_layer')
            layer = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal'), merge_mode='sum')(
                layer)
            layer = BatchNormalization()(layer)
            # lstm_3 = LSTM(256, return_sequences=True, kernel_initializer='he_normal',
            #               name='lstm_3')(layer)
            # lstm_4 = LSTM(256, return_sequences=True, kernel_initializer='he_normal',
            #               name='lstm_4', go_backwards=True)(layer)
            # # None,64,512
            # layer = concatenate([lstm_3, lstm_4], name='concat_layer')
            layer = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal'),
                                  merge_mode='concat')(layer)
            layer = BatchNormalization()(layer)
        else:
            # gru_1=GRU(256, return_sequences=True, kernel_initializer='he_normal',
            #           name='gru_1')(layer)
            # gru_2=GRU(256, return_sequences=True, kernel_initializer='he_normal',
            #           name='gru_2',go_backwards=True)(layer)
            # layer=add([gru_1,gru_2],name='gru_add_layer')
            layer = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='he_normal'), merge_mode='sum')(
                layer)
            layer = BatchNormalization()(layer)
            # gru_3=GRU(256, return_sequences=True, kernel_initializer='he_normal',
            #           name='gru_3')(layer)
            # gru_4=GRU(256, return_sequences=True, kernel_initializer='he_normal',
            #           name='gru_4',go_backwards=True)(layer)
            # layer=concatenate([gru_3,gru_4],name='gru_concatenate_layer')
            layer = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='he_normal'), merge_mode='concat')(
                layer)
            layer = BatchNormalization()(layer)

        layer=Dense(self.num_classes,kernel_initializer='he_normal',
                    name='dense_2')(layer)
        y_pred=Activation(activation='softmax')(layer)
        labels=Input(name='the_labels',shape=[self.max_text_len],dtype='float32')
        input_length=Input(name='input_length',shape=[1],dtype='int64')
        label_length=Input(name='label_length',shape=[1],dtype='int64')
        ctc_loss=Lambda(self.ctc_func,output_shape=(1,),name='ctc_loss')(
            [y_pred,labels,input_length,label_length]
        )
        if istraining:
            return Model(inputs=[inputs,labels,input_length,label_length],outputs=ctc_loss)
        else:
            return Model(inputs=[inputs],outputs=y_pred)
    def crnn_v3(self,istraining=True,isGRU=False):
        """
                固定输入尺寸是高：256，宽 64，训练与预测也必须改成这个尺寸
                :param istraining: check whether it is training mode
                :return:
                """
        K.set_image_data_format('channels_last')
        input_shape = (self.img_h, self.img_w, self.img_c)
        # 64,256,3
        inputs = Input(name='the_input',
                       shape=input_shape,
                       dtype='float32')
        # None,64,256,3
        layer = Conv2D(64, (3, 3), padding='same', name='conv1',
                       kernel_initializer='he_normal')(inputs)  # None,64,256,64
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling2D(pool_size=(2, 2), name='max1')(layer)  # None,32,128,64

        layer = Conv2D(128, (3, 3), padding='same', name='conv2',
                       kernel_initializer='he_normal')(layer)  # None,32,128,128
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling2D(pool_size=(2, 2), name='max2')(layer)  # None,16,64,128

        layer = Conv2D(256, (3, 3), padding='same', name='conv3',
                       kernel_initializer='he_normal')(layer)  # None,16,64,256
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(256, (3, 3), padding='same', name='conv4',
                       kernel_initializer='he_normal')(layer)  # None,16,64,256
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        # Attention!!!!!!!!
        layer = MaxPooling2D(pool_size=self.sp_pool_size, name='max3')(layer)  # None,8,64,256

        layer = Conv2D(512, (3, 3), padding='same', name='conv5',
                       kernel_initializer='he_normal')(layer)  # None,8,64,512
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(512, (3, 3), padding='same', name='conv6',
                       kernel_initializer='he_normal')(layer)  # None,8,64.512
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling2D(pool_size=self.sp_pool_size, name='max4')(layer)  # None,4,64.512

        layer = Conv2D(512, (2, 2), padding='same', name='conv7',
                       kernel_initializer='he_normal')(layer)  # None 64,4,512
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer=MaxPooling2D(pool_size=self.sp_pool_size,name='max5')(layer)

        layer = Conv2D(512, (2, 2), padding='same', name='conv8',
                       kernel_initializer='he_normal')(layer)  # None 64,4,512
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling2D(pool_size=self.sp_pool_size, name='max6')(layer)


        # CNN RNN
        # 由于我们固定了高度，使得输入一直是256，下采样之后，变成高度是64，我们根据每一行来建立sequeces，所以是feature map的高度
        layer = Reshape(target_shape=(self.img_h // 4, -1), name='reshape_layer')(layer)
        # # None,64,128
        # layer = Dense(128, activation='relu', kernel_initializer='he_normal',
        #               name='dense1')(layer)

        if isGRU == False:

            layer = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal'), merge_mode='sum')(
                layer)
            layer = BatchNormalization()(layer)

            layer = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal'),
                                  merge_mode='concat')(layer)
            layer = BatchNormalization()(layer)
        else:

            layer = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='he_normal'), merge_mode='sum')(
                layer)
            layer = BatchNormalization()(layer)

            layer = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='he_normal'), merge_mode='concat')(
                layer)
            layer = BatchNormalization()(layer)

        layer = Dense(self.num_classes, kernel_initializer='he_normal',
                      name='dense_2')(layer)
        y_pred = Activation(activation='softmax')(layer)
        labels = Input(name='the_labels', shape=[self.max_text_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        ctc_loss = Lambda(self.ctc_func, output_shape=(1,), name='ctc_loss')(
            [y_pred, labels, input_length, label_length]
        )
        if istraining:
            return Model(inputs=[inputs, labels, input_length, label_length], outputs=ctc_loss)
        else:
            return Model(inputs=[inputs], outputs=y_pred)
if __name__=="__main__":
    M=CRNN(
        num_classes=9116,
    ).crnn_v3(istraining=False,isGRU=False)
    M.summary()