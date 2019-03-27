from Recognition.CRNN.model import CRNN
from Recognition.DataGenerator import RecognitionGenerator
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau,TensorBoard,LearningRateScheduler
from keras import backend as K
from keras.metrics import categorical_accuracy
def mcor(y_true, y_pred):
    # matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def check(path):
    train=open(path,'r',encoding='utf-8').readlines()
    for train_line in train:
        flag=True
        train_line=train_line.split(' ')
        for i in train_line[1:]:
            for z in range(10):
                if i.startswith(str(z)) and i.endswith('g'):
                    print(path,i,train_line)
                    flag=False
                    raise ValueError(path,i,train_line)
                if i.startswith('i') and i.endswith(str(z)):
                    print(path,i,train_line)
                    flag = False
                    raise ValueError(path,i,train_line)

            if flag:
                print('check done')
def recognition_train(
        img_dir='D:\python_projects\data_new\data\\train_img_cut_new',
        labeled_file_path='D:\python_projects\data_new\data\\train_img_labels\labels.txt',
        chinese_set_path='D:\python_projects\data_new\data\chinese\chinese_all.txt',
        first_use=False,
        text_max_len=40
):
    train_data = RecognitionGenerator(
        img_dir=img_dir,
        labeled_file_path=labeled_file_path,
        chinese_set_path=chinese_set_path,
        first_use=first_use,
        mode='train',
        text_max_len=text_max_len
    )
    val_data = RecognitionGenerator(
        img_dir=img_dir,
        labeled_file_path=labeled_file_path,
        chinese_set_path=chinese_set_path,
        first_use=first_use,
        mode='val',
        text_max_len=text_max_len
    )
    check(train_data.train_label_path)
    check(train_data.val_label_path)
    check(train_data.test_label_path)
    model = CRNN(num_classes=train_data.number_classes, maxLen=train_data.max_text_len).crnn_v1(istraining=True,
                                                                                             isGRU=False)

    ada = Adadelta()
    LR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(filepath='model/crnn_v1_fixed_size_withoutGRU--{epoch:02d}--{val_loss:.3f}.hdf5',
                                 monitor='val_loss', save_best_only=False,
                                 verbose=1, mode='min', period=1)
    TB = TensorBoard(log_dir='logs', update_freq='epoch')
    model.compile(loss={'ctc_loss': lambda y_true, y_pred: y_pred}, optimizer=ada,
                  metrics=[ 'acc'])

    his=model.fit_generator(
        generator=train_data.generator(),
        steps_per_epoch=train_data.steps_per_epoch,
        validation_data=val_data.generator(),
        validation_steps=val_data.steps_per_epoch,
        epochs=100,
        initial_epoch=20,
        callbacks=[checkpoint, early_stop, LR, TB]
    )
    print(his)


from keras import backend as K
from PIL import Image
import cv2,numpy as np
from Recognition.CRNN.model import CRNN
import itertools, os, time

class predictor(object):
    def __init__(self,version):
        self.version=version
        self.test_label_path='info/test.txt'
        self.img_dir='D:\python_projects\data_new\data\\train_img_cut_new'
        self.chinese_path='D:\python_projects\data_new\data\chinese\chinese_all.txt'
        self.num2chinese={i:c for i,c in enumerate(open(self.chinese_path,'r',encoding='utf-8').readlines()[0].strip('\n').split(','))}
        self.model=CRNN(num_classes=len(self.num2chinese)+1).crnn_v1(istraining=False,isGRU=False) \
            if self.version=='v1' else CRNN.crnn_v2(istraining=False,isGRU=False)
        self.model.load_weights('model/model_withoutGRU--20--3.520.hdf5')
    def predict(self):
        def decode_label(out):
            # out : (1, 32, 42)
            out_best = list(np.argmax(out[0, :], axis=1))  # get max index -> len = 32
            out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
            outstr = ''
            for i in out_best:
                if i < len(self.num2chinese):
                    outstr += self.num2chinese[i]
                else:
                    pass
            return outstr
        test_images=open(self.test_label_path,'r',encoding='utf-8').readlines()
        for i in test_images:
            i=i.strip('\n').split(' ')
            image_name=i[0]
            true_label=i[1:]
            true_chinese=[]
            for j in i[1:]:
                true_chinese.append(self.num2chinese[int(j)])
            image=Image.open(os.path.join(self.img_dir,image_name))
            print(image.size)
            image=cv2.imread(os.path.join(self.img_dir,image_name))

            image = cv2.resize(image, (64, 256))
            cv2.imshow('test', image)
            image = (image / 255.0) * 2.0 - 1.0

            image = np.array([image])
            print(image.shape)
            out = self.model.predict(image)
            print(out.shape)
            print('predict ',decode_label(out))
            print('true label: ',true_chinese)
            cv2.waitKey(0)
            cv2.destroyAllWindows()





