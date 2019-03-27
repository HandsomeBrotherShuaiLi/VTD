import os,numpy as np
from keras.preprocessing import image
from PIL import Image
class RecognitionGenerator(object):
    def __init__(self,img_dir,chinese_set_path,labeled_file_path,
                 first_use=True,fixed_size=True,
                 train_batch_size=64,val_batch_size=50,text_max_len=40,test_batch_size=10,
                 mode='train',img_shape=(64,256,3),val_size=5000,test_size=100):
        """

        :param img_dir: 所有的图片的文件夹名
        :param chinese_set_path: 中文词汇集合
        :param labeled_file_path: 图片名称 对应的数字标签
        :param train_batch_size:
        :param val_batch_size:
        :param text_max_len:
        :param mode:
        :param img_shape: w=64   h=256
        :param val_size:
        :param test_size:
        """
        assert mode in ['train','val','test']
        self.img_dir=img_dir
        self.max_text_len=text_max_len
        self.img_shape=img_shape
        self.mode=mode
        self.batch_size={
            'train':train_batch_size,
            'val':val_batch_size,
            'test':test_batch_size
        }
        self.val_size=val_size
        self.test_size=test_size
        self.first_use=first_use
        self.labeled_file_path=labeled_file_path
        self.chinese_path=chinese_set_path
        self.chinese_dict=self.__load_chinese()
        self.number_classes=len(self.chinese_dict)+1

        self.fixed_size=fixed_size

        self.__init_shuffle()

    def __init_shuffle(self):
        self.train_label_path="info/train.txt"
        self.val_label_path='info/val.txt'
        self.test_label_path='info/test.txt'
        if self.first_use:
            img_labels = open(self.labeled_file_path, mode='r', encoding='utf-8').readlines()
            data=np.array(img_labels)
            np.random.shuffle(data)
            self.val_data=data[:self.val_size]
            self.test_data=data[self.val_size:self.val_size+self.test_size]
            self.train_data=data[self.val_size+self.test_size:]
            with open("info/train.txt",'w',encoding='utf-8') as f:
                f.writelines(self.train_data)
                f.close()
            with open('info/val.txt','w',encoding='utf-8') as f:
                f.writelines(self.val_data)
                f.close()
            with open('info/test.txt','w',encoding='utf-8') as f:
                f.writelines(self.test_data)
                f.close()
            print("随机选取{}个验证数据,保存到{},随机选取{}个测试数据,保存到{},还剩{}个训练数据,保存到{}".format(
                len(self.val_data),self.val_label_path,len(self.test_data),self.test_label_path,len(self.train_data),self.train_label_path
            ))
        E={
            'train':len(open(self.train_label_path,'r',encoding='utf-8').readlines())//self.batch_size['train'],
            'val':len(open(self.val_label_path,'r',encoding='utf-8').readlines())//self.batch_size['val'],
            'test':len(open(self.test_label_path,'r',encoding='utf-8').readlines())//self.batch_size['test']
        }
        self.steps_per_epoch=E[self.mode]

    def __load_chinese(self):
        chinese_set=open(self.chinese_path,mode='r',encoding='utf-8').readlines()[0].split(',')
        chinese_dict={i:c for i,c in enumerate(chinese_set)}
        return chinese_dict

    def generator(self):
        imgs_batch_list=[]
        labels_batch_list=[]
        input_batch_length=[]
        label_batch_length=[]
        count=0
        while True:
            with open('info/{}.txt'.format(self.mode),'r',encoding='utf-8') as f:
                for line in f:
                    line=line.strip('\n').split(' ')
                    img_name=line[0]
                    if self.mode in ['train','val']:
                        image_raw=Image.open(os.path.join(self.img_dir, img_name))
                        # width height
                        original_size=image_raw.size
                        # print('original size is (w,h) ',image_raw.size)
                        if self.img_shape[2]!=3:
                            image_raw=image_raw.convert('L')
                        if self.fixed_size==True:#改成宽64，高256
                            img=image_raw.resize((self.img_shape[0],self.img_shape[1]),Image.ANTIALIAS)
                        else:#改成按照宽的比例resize
                            img=image_raw.resize((64,int(original_size[1]*64/original_size[0])),Image.ANTIALIAS)
                        # print('resize后的shape is (w,h) ',img.size)
                        img=np.array(img)
                        img = (img / 255.0) * 2.0 - 1.0
                        # print('数组的 shape ',img.shape)
                    elif self.mode=='test':
                        pass
                    else:
                        raise ValueError('输入错误的模式')

                    label=np.ones([self.max_text_len])*self.number_classes
                    #nuber_classes 9116-> blank
                    label_length=len(line[1:])

                    label[0:label_length]=[int(i) for i in line[1:]]
                    # input_length=img.shape[0]//4
                    input_length=img.shape[0]//4
                    # print('input_length is ', input_length)
                    imgs_batch_list.append(img)
                    input_batch_length.append(input_length)
                    label_batch_length.append(label_length)
                    labels_batch_list.append(label)
                    count+=1

                    if count>=self.batch_size[self.mode]:
                        inputs={
                            'the_input':np.array(imgs_batch_list),
                            'the_labels':np.array(labels_batch_list),
                            'input_length':np.array(input_batch_length),
                            'label_length':np.array(label_batch_length)
                        }
                        outputs= {'ctc_loss': np.zeros([self.batch_size[self.mode]])}
                        # for i in inputs:
                        #     print(i,inputs[i].shape)
                        # print('读取完毕一个batch')
                        yield (inputs,outputs)
                        count=0
                        imgs_batch_list = []
                        labels_batch_list = []
                        input_batch_length = []
                        label_batch_length = []