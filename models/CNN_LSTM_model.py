# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_CNN_LSTM.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchvision.models as models

import sys
sys.path.append('../')
import os
import dataloader.VSLdataset as VSLdataset
import torch.optim as optim

"""
    Neural Network: CNN_LSTM
    Detail: the input crosss cnn model and LSTM model independly, then the result of both concat
"""


# https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/4
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
        self.lstm = nn.LSTM(cnn_out_size, self.hidden_dim, dropout=_dropout, num_layers=self.num_layers, batch_first=False)
        
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


        lstm_in = c_out.view(timesteps, batch_size, -1)
        
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        # linear
        cnn_lstm_out = self.hidden1_fc(torch.tanh(lstm_out))
        cnn_lstm_out = self.hidden2_fc(torch.tanh(cnn_lstm_out))
        # output
        logit = cnn_lstm_out
        #print(logit.shape)
        return logit       

# 测试一下输出的size        
def test_model():
    model = CLSTM(lstm_hidden_dim = 10, lstm_num_layers = 2, class_num=8)
    #batch, len, channel, width, height
    data = torch.randn(1, 10, 3, 224, 224)
    output = model(data)
    print(output,shape)

    
    
def train(model, num_epochs = 3):
    # === dataloader defination ===
    train_batch_size = 4
    valid_batch_size = 2
    test_batch_size = 1
    dataloaders = VSLdataset.create_dataloader_train_valid_test(train_batch_size, valid_batch_size, test_batch_size)
    train_dataloader = dataloaders['train']
    valid_dataloader = dataloaders['valid']
    test_dataloader = dataloaders['test']
    # =============================
    
    # === every n epochs print ===
    valid_epoch_step = 1
    test_epoch_step = 10
    # ============================
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    

    save_file = os.path.join('../saved_model', 'CLSTM.pth')
    for epoch in range(num_epochs):
        # training
        model.train()
        train_loss  = 0.0
        print('Train:')
        for index, (data, target) in enumerate(train_dataloader):
            #print('Epoch: ', epoch, '| Batch_index: ', index, '| data: ',data.shape, '| labels: ', target.shape)
            '''
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if index % 10  == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, index + 1, train_loss / 10))
                train_loss = 0.0
            '''

        if (epoch % valid_epoch_step == (valid_epoch_step -1)):
            # validation
            class_correct = list(0. for i in range(len(VSLdataset.class_name_to_id_)))
            class_total = list(0. for i in range(len(VSLdataset.class_name_to_id_)))
            class_name = list(VSLdataset.class_name_to_id_.keys())
            model.eval()
            print('Valid:')
            for index, (data, target) in enumerate(valid_dataloader):
                #print('Epoch: ', epoch, '| Batch_index: ', index, '| data: ',data.shape, '| labels: ', target.shape)
                data, target = data.to(device), target.to(device)
                output = model(data)
                #print(output.shape)
                _, predicted = torch.max(output, 1)
                #print(predicted.shape)
                c = (predicted == target).squeeze()
                print(c)
                for i in range(valid_batch_size):
                    try:
                        label = target[i]
                        class_correct[label] += c[i].item()
                    except:
                        label = target
                        class_correct[label] += c.item()
                    class_total[label] += 1
            for i in range(len(VSLdataset.class_name_to_id_)):
                print('Accuracy of %5s : %2d %%' % (
                    class_name[i], 100 * (class_correct[i] + 1) / (class_total[i] + 1)))
            
            # 每次 eval 都进行保存
            # save current model
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }, save_file)
            print("saved model.")
        # test
        '''
        model.eval()
        if (epoch%test_epoch_step == 0):
            valid_losses = []
            print('Test:')
            for index, batch in enumerate(test_dataloader):
                
                print('Epoch: ', epoch, '| Batch_index: ', index, '| data: ',data.shape, '| labels: ', target.shape)
                #loss = self._val_step(epoch, index, batch)
                #losses.append(loss)
        '''

if __name__=='__main__':
    #test_model()
    model = CLSTM(lstm_hidden_dim = 10, lstm_num_layers = 2, class_num=8)
    train(model, 100)
