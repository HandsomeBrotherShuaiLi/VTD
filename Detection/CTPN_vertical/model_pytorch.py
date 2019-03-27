import os,torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
class BasicConv(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride=1,padding=0,dilation=1,
                 groups=1,relu=True,bn=True,bias=True):
        nn.Module.__init__(self)
        self.out_channels=out_planes
        self.conv=nn.Conv2d(in_planes,out_planes,kernel_size=kernel_size,stride=stride,
                            padding=padding,dilation=dilation,groups=groups,bias=bias)
        self.bn=nn.BatchNorm2d(out_planes,momentum=0.01,affine=True) if bn else None
        self.relu=nn.ReLU(True) if relu else None

    def forward(self, x):
        x=self.conv(x)
        if self.bn is not None:
            x=self.bn(x)
        if self.relu is not None:
            x=self.relu(x)
        return x

class Proposal_model(nn.Module):
    def __init__(self,base_model_name=None,pretrained=False,k=10):
        nn.Module.__init__(self)
        if base_model_name!=None:
            if base_model_name.lower() == 'vgg16':
                self.base_model = models.vgg16(pretrained=pretrained)
            elif base_model_name.lower() == 'resnet50':
                self.base_model = models.resnet50(pretrained=pretrained)
            elif base_model_name.lower() == 'densenet121':
                self.base_model = models.densenet121(pretrained=pretrained)
            elif base_model_name.lower() == 'inception_v3':
                self.base_model = models.inception_v3(pretrained=pretrained)
        else:
            self.base_model = models.vgg16(pretrained=pretrained)
        layers=list(self.base_model.features)[:-1]
        self.base_layers=nn.Sequential(*layers)
        self.rpn=BasicConv(512,512,3,1,1,bn=True)
        self.brnn=nn.LSTM(512,128,bidirectional=True)
        self.lstm_fc=BasicConv(256,512,1,1,relu=True,bn=False)
        self.rpn_class=BasicConv(512,k*2,1,1,relu=False,bn=False)
        self.rpn_regress=BasicConv(512,k*2,1,1,relu=False,bn=False)
        self.k=k
    def forward(self, x):
        print('base layer之前的size',x.size())
        x=self.base_layers(x)
        print('base layer之后的size', x.size())
        x=self.rpn(x)
        print('rpn之后的size',x.size())
        #很明显这是把channel first 改成channel last ,但是读取的图片是这样吗？tensor的输出是这样吗？
        #这里待修改
        x1=x.permute(0,2,3,1).contiguous()
        print('x1 permute shape',x1.size())
        b=x1.size()
        #bs,h,w,c->bs*w,h,c
        x1=x1.view(b[0]*b[2],b[1],b[3])
        print('after viewing, x1 shape',x1.size())
        print('x1 is tensor? ',torch.is_tensor(x1))
        x2,_=self.brnn(x1)

        xsz=x.size()
        x3=x2.view(xsz[0],xsz[2],xsz[3],256)

        x3=x3.permute(0,3,1,2).contiguous()#channel first???
        x3=self.lstm_fc(x3)
        x=x3

        cls=self.rpn_class(x)
        regr=self.rpn_regress(x)

        cls=cls.permute(0,2,3,1).contiguous()
        regr=regr.permute(0,2,3,1).contiguous()
        cls = cls.view(cls.size(0), cls.size(1) * cls.size(2) * self.k, 2)
        regr = regr.view(regr.size(0), regr.size(1) * regr.size(2) * self.k, 2)
        return cls,regr

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m=Proposal_model(base_model_name='vgg16').to(device)
    summary(m,input_size=(3,1024,512),batch_size=1)

