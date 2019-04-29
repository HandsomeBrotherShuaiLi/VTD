from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from keras.optimizers import Adam,SGD,RMSprop
from Detection.AdvancedEAST.label import process_label_single_image,process_label_no_cfg
from Detection.AdvancedEAST import cfg
from Detection.AdvancedEAST.network import East
from Detection.Relation_Attention.relation_attention_network import RAN
from Detection.AdvancedEAST.losses import quad_loss
from Detection.AdvancedEAST.data_generator import gen,gen_plus
import numpy as np,os
from keras.applications.vgg16 import preprocess_input
from Detection.AdvancedEAST.predict import predict_new
class Detection:
    """
    要训练新的size 除了要修改img_size，也要修改 cfg task id
    """
    def __init__(self,computer_id,img_size,opt='adam'):
        assert computer_id in ['nio','youhua','jia']
        assert img_size in [256, 384, 512, 640, 736]
        self.img_size=img_size
        self.opt=opt
        self.train_task_id='SIZE'+str(img_size)
        if computer_id=='nio':
            self.project_model_dir='D:\py_projects\VTD\model\east_model\saved_model'
            self.csv_path='D:\py_projects\data_new\data_new\data\original_csv\concat_train.csv'
            self.data_dir='D:\py_projects\data_new\data_new\data'
            self.model_weights_path = 'D:\py_projects\VTD\model\east_model\epoch_weights\\'+opt+'_weights_%s.{epoch:03d}-{val_loss:.5f}.h5' \
                     % cfg.train_task_id
            self.saved_model_file_path = 'D:\py_projects\VTD\model\east_model\saved_model\\'+opt+'_east_model_%s.h5' % cfg.train_task_id
            self.saved_model_weights_file_path='D:\py_projects\VTD\model\east_model\saved_model\\'+opt+'_east_model_weights_%s.h5'\
                                % cfg.train_task_id
        elif computer_id =='youhua':
            self.project_model_dir='D:\python_projects\\vertical_text_detection\model\east_model\saved_model'
            self.csv_path = 'D:\python_projects\data_new\data\original_csv\concat_train.csv'
            self.data_dir = 'D:\python_projects\data_new\data'
            self.model_weights_path = 'D:\python_projects\\vertical_text_detection\model\east_model\epoch_weights\\'+opt+'_weights_%s.{epoch:03d}-{val_loss:.5f}.h5' \
                                     % cfg.train_task_id
            self.saved_model_file_path = 'D:\python_projects\\vertical_text_detection\model\east_model\saved_model\\'+opt+'_east_model_%s.h5' % cfg.train_task_id
            self.saved_model_weights_file_path = 'D:\python_projects\\vertical_text_detection\model\east_model\saved_model\\'+opt+'_east_model_weights_%s.h5' \
                                                % cfg.train_task_id
        else:
            self.project_model_dir='D:\\vertical_text_detection\model\east_model\saved_model'
            self.csv_path = 'D:\python_projects\data_new\data\original_csv\concat_train.csv'
            self.data_dir = 'D:\python_projects\data_new\data'
            self.model_weights_path = 'D:\\vertical_text_detection\model\east_model\epoch_weights\\' + opt + '_weights_%s.{epoch:03d}-{val_loss:.5f}.h5' \
                                     % cfg.train_task_id
            self.saved_model_file_path = 'D:\\vertical_text_detection\model\east_model\saved_model\\' + opt + '_east_model_%s.h5' % cfg.train_task_id
            self.saved_model_weights_file_path = 'D:\\vertical_text_detection\model\east_model\saved_model\\' + opt + '_east_model_weights_%s.h5' \
                                                % cfg.train_task_id
    def process(self):
        if self.img_size!=256:
            process_label_no_cfg(self.data_dir,self.img_size)

    def generator(self,is_val=False):
        img_h, img_w = self.img_size, self.img_size
        x = np.zeros((cfg.batch_size, img_h, img_w, cfg.num_channels), dtype=np.float32)
        pixel_num_h = img_h // cfg.pixel_size
        pixel_num_w = img_w // cfg.pixel_size
        y = np.zeros((cfg.batch_size, pixel_num_h, pixel_num_w, 7), dtype=np.float32)
        if is_val:
            with open(os.path.join(cfg.data_dir, 'val_SIZE256.txt'), 'r') as f_val:
                f_list = f_val.readlines()

        else:
            with open(os.path.join(cfg.data_dir, 'train_SIZE256.txt'), 'r') as f_train:
                f_list = f_train.readlines()

        while True:
            count = 0
            for i in range(len(f_list)):
                one_img = f_list[i]
                img_filename = one_img.strip('\n').split(',')[0]
                img,gt_file=process_label_single_image(img_filename,self.img_size)
                img=np.array(img)
                print(img.shape,gt_file.shape)
                x[count] = preprocess_input(img, mode='tf')
                y[count] = gt_file
                count += 1
                if count >= cfg.batch_size:
                    print(x.shape,y.shape)
                    yield x, y
                    count = 0
            np.random.shuffle(f_list)
    def train(self,model='east',load_weights=True,weights_path=None):
        if model=='east':
            model = East()
            network = model.east_network()
            network.summary()
        else:
            model=RAN()
            network=model.ran_network()
            network.summary()
        network.compile(loss=quad_loss, optimizer=Adam(lr=cfg.lr,decay=cfg.decay) if self.opt=='adam' else RMSprop(lr=cfg.lr,decay=cfg.decay))
        if load_weights:
            if weights_path==None:
                network.load_weights(os.path.join(self.project_model_dir,'east_model_weights_3T736.h5'))
            else:
                network.load_weights(weights_path)
        TB = TensorBoard(log_dir='logs',update_freq='batch')
        RL = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=0, verbose=1)
        ES = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=25, mode='min', verbose=1)
        CK = ModelCheckpoint(filepath=self.model_weights_path, save_best_only=True, save_weights_only=True, verbose=1)
        his = network.fit_generator(generator=gen_plus(shape=self.img_size,is_val=False),
                                    steps_per_epoch=cfg.steps_per_epoch,
                                    epochs=100,
                                    validation_data=gen_plus(shape=self.img_size,is_val=True),
                                    validation_steps=cfg.validation_steps,
                                    verbose=1,
                                    initial_epoch=cfg.initial_epoch,
                                    callbacks=[CK, TB, RL, ES]
                                    )
        print(his)
        network.save(cfg.saved_model_file_path)
        network.save_weights(cfg.saved_model_weights_file_path)

    def predict(self,predict_weight_path,mode=1):
        img_dir=os.path.join(cfg.data_dir,'test_img')
        predict_geo_txt_dir=os.path.join(cfg.data_dir,'predict_geo_txt')
        predict_img_dir=os.path.join(cfg.data_dir,'predict_img_dir')
        if not os.path.exists(predict_geo_txt_dir):
            os.mkdir(predict_geo_txt_dir)
        if not os.path.exists(predict_img_dir):
            os.mkdir(predict_img_dir)
        model=East().east_network()
        model.load_weights(predict_weight_path)
        if mode==1:
            for img_name in os.listdir(img_dir):
                if img_name.endswith('_bg.jpg'):
                    img_path = os.path.join(img_dir, img_name)
                    try:
                        f=predict_new(model, img_name, img_path, cfg.pixel_threshold, predict_img_dir,
                                    predict_geo_txt_dir)
                        with open('fail_imgs.txt','w',encoding='utf-8') as F:
                            F.write('\n'.join(f))
                    except Exception as e:
                        print(e)
if __name__=='__main__':
    d = Detection(
        computer_id='nio',
        img_size=384
    )
    d.process()
    # d.train(weights_path='D:\py_projects\VTD\model\east_model\saved_model\\adam_east_model_SIZE256.h5')
    # d.predict(predict_weight_path='D:\py_projects\VTD\model\east_model\saved_model\\adam_east_model_SIZE256.h5')