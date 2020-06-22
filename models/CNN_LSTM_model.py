# @Author : Xi Hu
# @Datetime : 2020/06/20 21:33
# @File : CNN_LSTM_model.py
# @Last Modify Time : 2020/06/20 21:33
# @Contact : chris_huxi@163.com

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
    
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter    

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
        c_display = c_in.view(batch_size, timesteps, C, H, W).permute(1, 0, 2, 3, 4).to("cpu")
        '''
        fig=plt.figure(figsize=(12, 6))
            
        fig.add_subplot(2,2,1)
        plt.imshow(c_display[0,0,:].view(c_display[0,0].shape[0], c_display[0,0].shape[1], c_display[0,0].shape[2]).permute(1, 2, 0))
            
        fig.add_subplot(2,2,2)
        plt.imshow(c_display[1,0,:].view(c_display[0,0].shape[0], c_display[0,0].shape[1], c_display[0,0].shape[2]).permute(1, 2, 0))
        
        fig.add_subplot(2,2,3)
        plt.imshow(c_display[0,1,:].view(c_display[0,0].shape[0], c_display[0,0].shape[1], c_display[0,0].shape[2]).permute(1, 2, 0))
            
        fig.add_subplot(2,2,4)
        plt.imshow(c_display[1,1,:].view(c_display[0,0].shape[0], c_display[0,0].shape[1], c_display[0,0].shape[2]).permute(1, 2, 0))
        
        plt.savefig('test2.png')
        '''
        
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


        lstm_in = c_out.view(batch_size, timesteps, -1).permute(1, 0, 2)
        
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

def load_checkpoint(model, checkpoint_PATH):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['model_state_dict'])
        print('loading checkpoint!')
    return model   
    
def train(model_in, num_epochs = 3, load_model = True):
    # === dataloader defination ===
    train_batch_size = 4
    valid_batch_size = 1
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


    save_file = os.path.join('../saved_model', 'CLSTM.pth')
    writer = SummaryWriter('../saved_model/tensorboard_log')

    if(load_model == True):
        model = load_checkpoint(model_in, save_file)
    else:
        model = model_in
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    loss_function = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=1,patience=2)

    # === runing no gpu ===
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # =====================

    
    for epoch in range(num_epochs):
        # training
        model.train()
        train_loss  = 0.0
        print('Train:')

        for index, (data, target) in enumerate(train_dataloader):
            #print('Epoch: ', epoch, '| Batch_index: ', index, '| data: ',data.shape, '| labels: ', target.shape)
            '''
            print(target[0])
            
            print(target[1])
            
            fig=plt.figure(figsize=(12, 6))
            
            fig.add_subplot(4,5,1)
            plt.imshow(data[0,0,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(4,5,2)
            plt.imshow(data[0,1,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(4,5,3)
            plt.imshow(data[0,2,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(4,5,4)
            plt.imshow(data[0,3,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))

            fig.add_subplot(4,5,5)
            plt.imshow(data[0,4,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(4,5,6)
            plt.imshow(data[0,5,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(4,5,7)
            plt.imshow(data[0,6,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(4,5,8)
            plt.imshow(data[0,7,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(4,5,9)
            plt.imshow(data[0,8,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))

            fig.add_subplot(4,5,10)
            plt.imshow(data[0,9,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))

            
            fig.add_subplot(4,5,11)
            plt.imshow(data[1,0,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(4,5,12)
            plt.imshow(data[1,1,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(4,5,13)
            plt.imshow(data[1,2,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(4,5,14)
            plt.imshow(data[1,3,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))

            fig.add_subplot(4,5,15)
            plt.imshow(data[1,4,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(4,5,16)
            plt.imshow(data[1,5,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(4,5,17)
            plt.imshow(data[1,6,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(4,5,18)
            plt.imshow(data[1,7,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))
            
            fig.add_subplot(4,5,19)
            plt.imshow(data[1,8,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))

            fig.add_subplot(4,5,20)
            plt.imshow(data[1,9,:].view(data[0,0].shape[0], data[0,0].shape[1], data[0,0].shape[2]).permute(1, 2, 0))

            plt.savefig('test.png')
            #break
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
                writer.add_scalar('Train/Loss', train_loss / 10, epoch * (len(train_dataloader)) + index + 1)
                writer.flush()
                train_loss = 0.0

        if (epoch % valid_epoch_step == (valid_epoch_step -1)):
            # validation
            class_correct = list(0. for i in range(len(VSLdataset.class_name_to_id_)))
            class_total = list(0. for i in range(len(VSLdataset.class_name_to_id_)))
            class_name = list(VSLdataset.class_name_to_id_.keys())
            model.eval()
            print('Valid:')
            loss_eval = 0.0
            loss_for_display = 0.0
            for index_eval, (data_eval, target_eval) in enumerate(valid_dataloader):
                data_eval, target_eval = data_eval.to(device), target_eval.to(device)
                output_eval = model(data_eval)
                loss_i = loss_function(output_eval, target_eval).item()
                loss_for_display += loss_i
                loss_eval += loss_i
                
                _, predicted = torch.max(output_eval, 1)

                c = (predicted == target_eval).squeeze()
                for i in range(valid_batch_size):
                    try:
                        label = target_eval[i]
                        class_correct[label] += c[i].item()
                    except:
                        label = target_eval
                        class_correct[label] += c.item()
                    class_total[label] += 1
                if index_eval % 10  == 9:    # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, index_eval + 1, loss_for_display / 10))
                    writer.add_scalar('Valid/Loss ', loss_for_display / 10, epoch*(len(valid_dataloader)) + index_eval + 1)
                    writer.flush()
                    loss_for_display = 0.0

            for i in range(len(VSLdataset.class_name_to_id_)):
                accuracy = 100 * (class_correct[i] + 1) / (class_total[i] + 1)
                print('Accuracy of %5s : %2d %%' % (
                    class_name[i], accuracy))
                # Record loss and accuracy from the test run into the writer
                writer.add_scalar('Valid/Accuracy ' + str(class_name[i]), accuracy, epoch)
                writer.flush()

            scheduler.step(loss_eval/len(valid_dataloader))
            loss_eval = 0.0

            # 每次 eval 都进行保存
            # save current model
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }, save_file)
            print("saved model.")
        # test ==> TODO
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
    model = CLSTM(lstm_hidden_dim = 128, lstm_num_layers = 2, class_num=8)        
    train(model, 100, False)
