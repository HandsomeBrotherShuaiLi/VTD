import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from Detection.CTPN_vertical.model_pytorch import Proposal_model
from Detection.CTPN_vertical.utils import MyDataSet
import time
torch.backends.cudnn.deterministic=True
random_seed = 2019
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)


class RPN_REGR_Loss(nn.Module):
    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, input, target):
        '''
        smooth L1 loss
        :param input:y_preds
        :param target: y_true
        :return:
        '''
        try:
            cls = target[0, :, 0]
            regr = target[0, :, 1:3]
            regr_keep = (cls == 1).nonzero()[:, 0]
            regr_true = regr[regr_keep]
            regr_pred = input[0][regr_keep]
            diff = torch.abs(regr_true - regr_pred)
            less_one = (diff < 1.0 / self.sigma).float()
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1 - less_one) * (diff - 0.5 / self.sigma)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            print('RPN_REGR_Loss Exception:', e)
            # print(input, target)
            loss = torch.tensor(0.0)

        return loss.to(self.device)


class RPN_CLS_Loss(nn.Module):
    def __init__(self, device, k):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device
        self.k = k

    def forward(self, input, target):
        y_true = target[0][0]
        cls_keep = (y_true != -1).nonzero()[:, 0]
        cls_true = y_true[cls_keep].long()
        cls_pred = input[0][cls_keep]
        loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1),
                          cls_true)  # original is sparse_softmax_cross_entropy_with_logits
        # loss = nn.BCEWithLogitsLoss()(cls_pred[:,0], cls_true.float())  # 18-12-8
        loss = torch.clamp(torch.mean(loss), 0, self.k) if loss.numel() > 0 else torch.tensor(0.0)
        return loss.to(self.device)


class ctpn_trainer(object):
    def __init__(self, image_dir, label_dir, checkpoint_dir, k=10, base_model_name='vgg16',
                 pretrained=False, lr=1e-3, epochs=100, initial_epoch=0, trained_weight=None, optimizer='Adam',
                 iou_positive=0.7, iou_negative=0.3, iou_select=0.7, rpn_positive_num=150, rpn_total_num=300
                 ):
        """

        :param image_dir:
        :param checkpoint_dir:
        :param lr:
        :param epochs:
        :param initial_epoch:
        :param trained_weight:
        """
        self.image_dir = image_dir
        self.lr = lr
        self.epochs = epochs
        self.initial_epoch = initial_epoch
        self.checkpoint_dir = checkpoint_dir
        self.trained_weight = trained_weight
        self.optimizer = optimizer
        self.k = k
        self.pretrained = pretrained
        self.base_model_name = base_model_name
        self.label_dir = label_dir
        self.iou_positive = iou_positive
        self.iou_negative = iou_negative
        self.iou_select = iou_select
        self.rpn_total_num = rpn_total_num
        self.rpn_positive_num = rpn_positive_num

    def save_checkpoint(self, state, epoch, loss_cls,
                        loss_regr, loss):
        ck_path = os.path.join(self.checkpoint_dir,
                               'epoch_{}-val_loss_cls_{:.5f}-val_loss_regr_{:.5f}-val_loss_{:.5f}.pth'.format(epoch,
                                                                                                              loss_cls,
                                                                                                              loss_regr,
                                                                                                              loss))
        torch.save(state, ck_path)
        print('saved checkpoint @ {}'.format(ck_path))

    def train(self, batch_size=80, validation_number=2000, val_batch_size=50, image_shape=None):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = Proposal_model(k=self.k, base_model_name=self.base_model_name, pretrained=self.pretrained)
        model.to(device)
        if self.trained_weight != None:
            print('load trained weight...')
            trained_parameters = torch.load(self.trained_weight, map_location=device)
            model.load_state_dict(trained_parameters['model_state_dict'])
            self.initial_epoch = trained_parameters['epoch']
        parameter_update = model.parameters()
        if self.optimizer.lower() == 'adam':
            optimizer = optim.Adam(params=parameter_update, lr=self.lr)
        elif self.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(params=parameter_update, lr=self.lr, momentum=0.9)
        elif self.optimizer.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params=parameter_update, lr=self.lr, momentum=0.9)
        else:
            raise IndexError('输入的optimizer错误')

        LOSS_CLS = RPN_CLS_Loss(device=device, k=self.k)
        LOSS_REGR = RPN_REGR_Loss(device=device)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        data = MyDataSet(image_dir=self.image_dir, label_dir=self.label_dir, image_shape=image_shape,
                         base_model_name=self.base_model_name,
                         iou_positive=self.iou_positive, iou_negative=self.iou_negative, iou_select=self.iou_select,
                         rpn_positive_num=self.rpn_positive_num, rpn_total_num=self.rpn_total_num)
        if image_shape == None:
            batch_size = 1
        all_index = list(range(len(data)))
        validation_index = np.random.choice(all_index, size=validation_number, replace=False)
        train_index = [i for i in all_index if i not in validation_index]
        print(len(validation_index))
        train_sampler = SubsetRandomSampler(train_index)
        validation_sampler = SubsetRandomSampler(validation_index)
        train_data = DataLoader(data, batch_size=batch_size, num_workers=10, sampler=train_sampler)
        validation_data = DataLoader(data, batch_size=val_batch_size, num_workers=10, sampler=validation_sampler)
        data_lengths = {'train': len(train_data), 'val': len(validation_data)}
        print('*' * 80)
        print(data_lengths)
        print('data loading finished')
        print('*' * 80)
        for epoch in range(self.initial_epoch + 1, self.initial_epoch + self.epochs + 1):
            print('*' * 80)
            print(time.ctime())
            start_time = time.time()
            print('Time:{} Epoch {}/{} starting...'.format(time.ctime(), epoch, self.epochs))

            train_epoch_loss = 0
            train_epoch_regr_loss = 0
            train_epoch_cls_loss = 0
            val_epoch_loss = 0
            val_epoch_regr_loss = 0
            val_epoch_cls_loss = 0
            for mode in ['train', 'val']:
                if mode == 'train':
                    epoch_step = data_lengths['train']
                    model.train(mode=True)
                    for batch_i, (imgs, clss, regrs) in enumerate(train_data):
                        imgs = imgs.cuda()
                        clss = clss.to(device)
                        regrs = regrs.to(device)

                        optimizer.zero_grad()

                        out_cls, out_regr = model.forward(imgs)
                        loss_cls, loss_regr = LOSS_CLS.forward(input=out_cls, target=clss), LOSS_REGR.forward(
                            input=out_regr, target=regrs)
                        # print(loss_cls,loss_regr)
                        loss = loss_cls + loss_regr

                        loss.backward()
                        optimizer.step()

                        train_epoch_cls_loss += loss_cls.item()
                        train_epoch_regr_loss += loss_regr.item()
                        train_epoch_loss += loss.item()

                        print(
                            '{} Epoch {}/{} Batch:{}/{} train---->loss_cls:{:.5f} loss_regr:{:.5f} loss:{:.5f}'.format(
                                time.ctime(), epoch, self.epochs, batch_i + 1, epoch_step, train_epoch_cls_loss/(batch_i + 1),
                                train_epoch_regr_loss/(batch_i + 1), train_epoch_loss/(batch_i + 1)))
                else:
                    with torch.no_grad():

                        model.eval()
                        print('start eval the model...')
                        for batch_i, (imgs, clss, regrs) in enumerate(validation_data):
                            imgs = imgs.cuda()
                            clss = clss.to(device)
                            regrs = regrs.to(device)

                            out_cls, out_regr = model.forward(imgs)
                            loss_cls, loss_regr = LOSS_CLS.forward(input=out_cls, target=clss), LOSS_REGR.forward(
                                input=out_regr, target=regrs)
                            loss = loss_cls.item() + loss_regr.item()

                            val_epoch_cls_loss += loss_cls.item()
                            val_epoch_regr_loss += loss_regr.item()
                            val_epoch_loss += loss
                        lr_scheduler.step(val_epoch_loss / data_lengths['val'], epoch=epoch)
                        print('\n\nEpoch:{}/{}\n'
                              'train loss:{:.5f} train cls loss:{:.5f} train regr loss:{:.5f}\n'
                              'val loss:{:.5f} val cls loss:{:.5f} val regr loss:{:.5f}\n'.format(
                            epoch, self.epochs, train_epoch_loss / data_lengths['train'],
                                                train_epoch_cls_loss / data_lengths['train'],
                                                train_epoch_regr_loss / data_lengths['train'],
                                                val_epoch_loss / data_lengths['val'],
                                                val_epoch_cls_loss / data_lengths['val'],
                                                val_epoch_regr_loss / data_lengths['val']
                        ))

            self.save_checkpoint(
                {'model_state_dict': model.state_dict(),
                 'epoch': epoch,
                 'optimizer': optimizer.state_dict()},
                loss=val_epoch_loss / data_lengths['val'],
                loss_regr=val_epoch_regr_loss / data_lengths['val'],
                loss_cls=val_epoch_cls_loss / data_lengths['val'],
                epoch=epoch
            )

            print(time.ctime())
            print('用时{}s'.format(time.time() - start_time))
            print('*' * 80)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
if __name__=='__main__':
    ctpn_trainer(image_dir='D:\py_projects\data_new\data_new\data\\train_img',
                 label_dir='D:\py_projects\data_new\data_new\data\\annotation',
                 checkpoint_dir='D:\py_projects\\vertical_text_detection\model\ctpn_v1').train(image_shape=(512,1024))












