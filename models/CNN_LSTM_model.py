# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_CNN_LSTM.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models

"""
    Neural Network: CNN_LSTM
    Detail: the input crosss cnn model and LSTM model independly, then the result of both concat
"""

class MyResnet50(models.resnet.ResNet):
    def __init__(self, pretrained=False):
        # Pass default resnet50 arguments to super init
        # https://github.com/pytorch/vision/blob/e130c6cca88160b6bf7fea9b8bc251601a1a75c5/torchvision/models/resnet.py#L260
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(models.resnet50(pretrained=True).state_dict())

        
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

'''
model = MyResnet50(pretrained=True)
x = torch.randn(2, 3, 224, 224)
output = model(x)
'''

#https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/4
# https://blog.csdn.net/shanglianlm/article/details/86376627 resnet 用法
class CLSTM(models.resnet.ResNet):
    def __init__(self, lstm_hidden_dim, lstm_num_layers, class_num, pretrained=True):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3])
        self.hidden_dim = lstm_hidden_dim
        self.num_layers = lstm_num_layers
        self.image_width = 224
        self.image_height = 224
        self.class_num = class_num
        
        _dropout = 0.7
        
        if pretrained:
            self.load_state_dict(models.resnet50(pretrained=True).state_dict())
        
        cnn_out_size = 2048
        self.lstm = nn.LSTM(cnn_out_size, self.hidden_dim, dropout=_dropout, num_layers=self.num_layers, batch_first=True)
        
        # linear
        self.hidden1_fc = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2_fc = nn.Linear(self.hidden_dim // 2, self.class_num)
        # dropout
        self.dropout = nn.Dropout(p=_dropout)
        
    def forward(self, x):
        # size: batch, len, channel, width, height
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        # ResNet:
        cnn_x = self.conv1(c_in)
        cnn_x = self.bn1(cnn_x)
        cnn_x = self.relu(cnn_x)
        cnn_x = self.maxpool(cnn_x)

        cnn_x = self.layer1(cnn_x)
        cnn_x = self.layer2(cnn_x)
        cnn_x = self.layer3(cnn_x)
        cnn_x = self.layer4(cnn_x)

        cnn_x = self.avgpool(cnn_x)
        c_out = torch.flatten(cnn_x, 1) # batch*len, 2048
        print(c_out)

        lstm_in = c_out.view(batch_size, timesteps, -1)
        
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        # linear
        cnn_lstm_out = self.hidden1_fc(torch.tanh(lstm_out))
        cnn_lstm_out = self.hidden2_fc(torch.tanh(cnn_lstm_out))
        # output
        logit = cnn_lstm_out

        return logit       
        
def test_model():
    model = CLSTM(lstm_hidden_dim = 10, lstm_num_layers = 2, class_num=8)
    #batch, len, channel, width, height
    data = torch.randn(1, 10, 3, 224, 224)
    output = model(data)
    print(output)
    
def train():
    pass
'''
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data = np.expand_dims(data, axis=1)
        data = torch.FloatTensor(data)
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0])) 
'''

if __name__=='__main__':
    test_model()
