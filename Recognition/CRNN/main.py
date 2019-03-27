from Recognition.DataGenerator import RecognitionGenerator
from Recognition.CRNN.model import CRNN
from Recognition.CRNN.utils import check
from keras.optimizers import Adadelta,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau,TensorBoard,LearningRateScheduler
from keras import backend as K
import tkinter as tk
import os,time
class Recognition(object):
    def __init__(self,img_dir,labeled_file_path,chinese_set_path,epoch=100,
                 version='v1',fixed_size=True,isGRU=False,set_img_size=(64,256,3),
                 first_use=False,text_max_len=35,train_batch_size=80,val_batch_size=50):
        """

        :param img_dir: 所有图片集合的文件夹
        :param labeled_file_path: 标签文本文件路径
        :param chinese_set_path: 中文汉字路径
        :param first_use: 是否是第一次使用
        :param text_max_len: 任意一幅字画中的一个竖行，最多的字数个数，default is 40
        :param version: version v1 必须是fixed_size=True,如果version v2,那么训练时候是不是固定size,还不知道
        """

        self.epoch=epoch
        self.img_dir=img_dir
        self.labeled_file_path=labeled_file_path
        self.chinese_set_path=chinese_set_path
        self.first_use=first_use
        self.text_max_len=text_max_len
        self.num2chinse={i:c for i,c in enumerate(open(self.chinese_set_path,'r',
                                                       encoding='utf-8').readlines()[0].strip('\n').split(','))}
        self.num_classes=len(self.num2chinse)+1
        self.version=version
        assert self.version in ['v1', 'v2','v3']
        self.fixed_size=fixed_size
        self.isGRU=isGRU
        self.train_bs=train_batch_size
        self.val_bs=val_batch_size
        self.set_image_shape=set_img_size
        if self.version=='v1' and self.fixed_size==False:
            raise ValueError('if you choose v1 then argument fixed_size must be True')

    def train(self):
        self.train_generator = RecognitionGenerator(
            img_dir=self.img_dir, labeled_file_path=self.labeled_file_path,
            chinese_set_path=self.chinese_set_path, first_use=self.first_use,
            fixed_size=self.fixed_size, train_batch_size=self.train_bs, val_batch_size=self.val_bs,
            img_shape=self.set_image_shape, text_max_len=self.text_max_len, mode='train'
        )
        self.val_generator = RecognitionGenerator(
            img_dir=self.img_dir, labeled_file_path=self.labeled_file_path,
            chinese_set_path=self.chinese_set_path, first_use=self.first_use,
            fixed_size=self.fixed_size, train_batch_size=self.train_bs, val_batch_size=self.val_bs,
            img_shape=self.set_image_shape, text_max_len=self.text_max_len, mode='val'
        )

        if self.train_generator.number_classes != self.num_classes or self.val_generator.number_classes!=self.num_classes:
            raise ValueError('number class error!')
        # check(self.train_generator.train_label_path)
        # check(self.train_generator.val_label_path)
        # check(self.train_generator.test_label_path)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!check done!!!!!!!!!!!!!!!!!!!!!!')
        dirs='model/crnn_{}_fixed_size_{}_isGRU_{}'.format(str(self.version),str(self.fixed_size),str(self.isGRU))
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        ada = Adadelta()
        sgd=SGD(lr=0.0001)
        LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0,verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, mode='min', verbose=1)
        checkpoint = ModelCheckpoint(
                                     filepath=os.path.join(dirs,'crnn--{epoch:02d}--{val_loss:.5f}--{val_acc:.5f}.hdf5'),
                                     monitor='val_loss', save_best_only=False,
                                     verbose=1, mode='min', period=1)
        TB = TensorBoard(log_dir='logs')
        model=CRNN(num_classes=self.num_classes,maxLen=self.text_max_len)
        if self.version=='v1':
            model=model.crnn_v1(istraining=True,isGRU=self.isGRU)
        elif self.version=='v2':
            model=model.crnn_v2(istraining=True,isGRU=self.isGRU)
        else:
            model=model.crnn_v3(istraining=True,isGRU=self.isGRU)
        model.compile(
            loss={'ctc_loss': lambda y_true, y_pred: y_pred}, optimizer=sgd,
            metrics=['acc']
        )
        model.load_weights('model/crnn_v1_fixed_size_True_isGRU_False/crnn_--39--1.070.hdf5')
        his = model.fit_generator(
            generator=self.train_generator.generator(),
            steps_per_epoch=self.train_generator.steps_per_epoch,
            validation_data=self.val_generator.generator(),
            validation_steps=self.val_generator.steps_per_epoch,
            epochs=self.epoch,
            initial_epoch=39,
            callbacks=[checkpoint, early_stop, LR, TB]
        )
        print(his)

    def predict(self,predict_model_version,model_path,mode=1):
        """
        predict the untrained images mode 1-> visualization
        2-> return int and string with visualization
        :return:
        """
        import itertools,numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        def decode_label(out):
            out_best = list(np.argmax(out[0, :], axis=1))  # get max index -> len = 32
            out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
            outstr = ''
            for i in out_best:
                if i < len(self.num2chinse):
                    outstr += self.num2chinse[i]
                else:
                    pass
            return outstr
        model=CRNN(
            num_classes=self.num_classes,
            maxLen=self.text_max_len
        )
        # mapping={'v1':model.crnn_v1(istraining=False,isGRU=self.isGRU),
        #          'v2':model.crnn_v2(istraining=False,isGRU=self.isGRU)}
        # predictor=mapping[predict_model_version].load_weights(model_path)
        if predict_model_version=='v1':
            model=model.crnn_v1(istraining=False,isGRU=False)
            model.load_weights(model_path)
        else:
            model=model.crnn_v2(istraining=False,isGRU=False)
            model.load_weights(model_path)
        print('load model done')
        test_image_list=open('info/test.txt','r',encoding='utf-8').readlines()
        test_char_acc=0
        test_sentence_acc=0
        all_char_len=0
        if mode==1:
            for test_image in test_image_list:
                test_image=test_image.strip('\n').split(' ')
                image_name=test_image[0]
                true_label=test_image[1:]
                true_chinese=''.join([self.num2chinse[int(i)] for i in true_label])
                img=Image.open(os.path.join(self.img_dir,image_name))
                print('original size',img.size)

                # plt.figure(image_name+' original size map')
                # plt.imshow(img)
                # plt.show()
                if predict_model_version=='v1':
                    """
                    v1 必须把图片变成高256，宽64
                    """

                    img=img.resize((64,256),Image.ANTIALIAS)
                    # plt.figure(image_name + ' resized map')
                    # plt.imshow(img)
                    # plt.show()
                    img=np.array(img)
                    img=(img/ 255.0) * 2.0 - 1.0
                    image = np.array([img])
                    print(image.shape)
                    out=model.predict(image)
                    pred=decode_label(out)
                    if pred==true_chinese:
                        test_sentence_acc+=1
                        test_char_acc+=len(pred)
                        all_char_len+=len(pred)
                    else:
                        all_char_len += len(true_chinese)
                        for i in range(len(true_chinese)):
                            try:
                                if pred[i]==true_chinese[i]:
                                    test_char_acc+=1
                            except:
                                pass

                    print('prediction',pred)
                    print('true content',true_chinese)

                else:
                    raw_size=img.size

                    # img = img.resize((64, int(64*raw_size[1]/raw_size[0])), Image.ANTIALIAS)
                    img = img.resize((64, 256), Image.ANTIALIAS)
                    # plt.figure(image_name + ' resized map')
                    # plt.imshow(img)
                    # plt.show()
                    img = np.array(img)
                    img = (img / 255.0) * 2.0 - 1.0
                    image = np.array([img])
                    print(image.shape)
                    out = model.predict(image)
                    pred=decode_label(out)
                    if pred==true_chinese:
                        test_sentence_acc+=1
                        test_char_acc+=len(pred)
                        all_char_len+=len(pred)
                    else:
                        all_char_len += len(true_chinese)
                        for i in range(len(true_chinese)):
                            try:
                                if pred[i]==true_chinese[i]:
                                    test_char_acc+=1
                            except:
                                pass
                    print('prediction',pred )
                    print('true content', true_chinese)

            print('sentence acc:',test_sentence_acc/len(test_image_list))
            print('char acc:',test_char_acc/all_char_len)





















