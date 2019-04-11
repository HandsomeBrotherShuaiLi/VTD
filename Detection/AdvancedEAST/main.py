from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from keras.optimizers import Adam,SGD,RMSprop
from Detection.AdvancedEAST.label import process_label_single_image
from Detection.AdvancedEAST import cfg
from Detection.AdvancedEAST.network import East
from Detection.AdvancedEAST.losses import quad_loss
from Detection.AdvancedEAST.data_generator import gen,gen_plus
import numpy as np,os
from keras.applications.vgg16 import preprocess_input
class Detection:
    def __init__(self,computer_id,img_size,opt='adam'):
        assert computer_id in ['nio','youhua','jia']
        assert img_size in [256, 384, 512, 640, 736]
        self.img_size=img_size
        self.opt=opt
        cfg.train_task_id='SIZE'+str(img_size)
        if computer_id=='nio':
            self.project_model_dir='D:\py_projects\VTD\model\east_model\saved_model'
            cfg.csv_path='D:\py_projects\data_new\data_new\data\original_csv\concat_train.csv'
            cfg.data_dir='D:\py_projects\data_new\data_new\data'
            cfg.model_weights_path = 'D:\py_projects\VTD\model\east_model\epoch_weights\\'+opt+'_weights_%s.{epoch:03d}-{val_loss:.5f}.h5' \
                     % cfg.train_task_id
            cfg.saved_model_file_path = 'D:\py_projects\VTD\model\east_model\saved_model\\'+opt+'_east_model_%s.h5' % cfg.train_task_id
            cfg.saved_model_weights_file_path='D:\py_projects\VTD\model\east_model\saved_model\\'+opt+'_east_model_weights_%s.h5'\
                                % cfg.train_task_id
        elif computer_id =='youhua':
            self.project_model_dir='D:\python_projects\\vertical_text_detection\model\east_model\saved_model'
            cfg.csv_path = 'D:\python_projects\data_new\data\original_csv\concat_train.csv'
            cfg.data_dir = 'D:\python_projects\data_new\data'
            cfg.model_weights_path = 'D:\python_projects\\vertical_text_detection\model\east_model\epoch_weights\\'+opt+'_weights_%s.{epoch:03d}-{val_loss:.5f}.h5' \
                                     % cfg.train_task_id
            cfg.saved_model_file_path = 'D:\python_projects\\vertical_text_detection\model\east_model\saved_model\\'+opt+'_east_model_%s.h5' % cfg.train_task_id
            cfg.saved_model_weights_file_path = 'D:\python_projects\\vertical_text_detection\model\east_model\saved_model\\'+opt+'_east_model_weights_%s.h5' \
                                                % cfg.train_task_id
        else:
            self.project_model_dir='D:\\vertical_text_detection\model\east_model\saved_model'
            cfg.csv_path = 'D:\python_projects\data_new\data\original_csv\concat_train.csv'
            cfg.data_dir = 'D:\python_projects\data_new\data'
            cfg.model_weights_path = 'D:\\vertical_text_detection\model\east_model\epoch_weights\\' + opt + '_weights_%s.{epoch:03d}-{val_loss:.5f}.h5' \
                                     % cfg.train_task_id
            cfg.saved_model_file_path = 'D:\\vertical_text_detection\model\east_model\saved_model\\' + opt + '_east_model_%s.h5' % cfg.train_task_id
            cfg.saved_model_weights_file_path = 'D:\\vertical_text_detection\model\east_model\saved_model\\' + opt + '_east_model_weights_%s.h5' \
                                                % cfg.train_task_id
    def generator(self,is_val=False):
        img_h, img_w = cfg.max_train_img_size, cfg.max_train_img_size
        x = np.zeros((cfg.batch_size, img_h, img_w, cfg.num_channels), dtype=np.float32)
        pixel_num_h = img_h // cfg.pixel_size
        pixel_num_w = img_w // cfg.pixel_size
        y = np.zeros((cfg.batch_size, pixel_num_h, pixel_num_w, 7), dtype=np.float32)
        if is_val:
            with open(os.path.join(cfg.data_dir, cfg.val_fname), 'r') as f_val:
                f_list = f_val.readlines()

        else:
            with open(os.path.join(cfg.data_dir, cfg.train_fname), 'r') as f_train:
                f_list = f_train.readlines()

        while True:
            count = 0
            for i in range(len(f_list)):
                one_img = f_list[i]
                img_filename = one_img.strip('\n').split(',')[0]
                img,gt_file=process_label_single_image(img_filename)
                img=np.array(img)
                x[count] = preprocess_input(img, mode='tf')
                y[count] = gt_file
                count += 1
                if count >= cfg.batch_size:
                    yield x, y
                    count = 0
            np.random.shuffle(f_list)
    def train(self,load_weights=True,weights_path=None):
        east = East()
        east_network = east.east_network()
        east_network.summary()
        east_network.compile(loss=quad_loss, optimizer=Adam(lr=cfg.lr,decay=cfg.decay) if self.opt=='adam' else RMSprop(lr=cfg.lr,decay=cfg.decay))
        if load_weights:
            if weights_path==None:
                east_network.load_weights(os.path.join(self.project_model_dir,'east_model_weights_3T736.h5'))
            else:
                east_network.load_weights(weights_path)
        TB = TensorBoard(log_dir='logs')
        RL = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0, verbose=1)
        ES = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='min', verbose=1)
        CK = ModelCheckpoint(filepath=cfg.model_weights_path, save_best_only=True, save_weights_only=True, verbose=1)
        if self.img_size==256:
            his = east_network.fit_generator(generator=gen_plus(),
                                             steps_per_epoch=cfg.steps_per_epoch,
                                             epochs=100,
                                             validation_data=gen_plus(is_val=True),
                                             validation_steps=cfg.validation_steps,
                                             verbose=1,
                                             initial_epoch=cfg.initial_epoch,
                                             callbacks=[CK, TB, RL, ES]
                                             )
        else:
            his = east_network.fit_generator(generator=self.generator(is_val=False),
                                             steps_per_epoch=cfg.steps_per_epoch,
                                             epochs=100,
                                             validation_data=self.generator(is_val=True),
                                             validation_steps=cfg.validation_steps,
                                             verbose=1,
                                             initial_epoch=cfg.initial_epoch,
                                             callbacks=[CK, TB, RL, ES]
                                             )
        print(his)
        east_network.save(cfg.saved_model_file_path)
        east_network.save_weights(cfg.saved_model_weights_file_path)

if __name__=='__main__':
    d = Detection(
        computer_id='nio',
        img_size=384
    )
    d.train(weights_path='D:\py_projects\VTD\model\east_model\epoch_weights\weights_SIZE256.007-0.17356.h5')