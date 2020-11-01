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
#import dataloader.VSLdataset_long as VSLdataset
import dataloader.VSLdataset as VSLdataset

import torch.optim as optim
    
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter    

import evaluate

from torch2trt import torch2trt

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

from torch2trt import TRTModule
import time


"""
    Neural Network: CNN_LSTM
    Detail: the input crosss cnn model and LSTM model independly, then the result of both concat
"""

class CLSTM(models.resnet.ResNet):
    def __init__(self, lstm_hidden_dim, lstm_num_layers, class_num, pretrained=True):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3]) # 50
        #super().__init__(models.resnet.Bottleneck, [3, 4, 23, 3]) # 101
        #super().__init__(models.resnet.BasicBlock, [2, 2, 2, 2]) # 18

        self.hidden_dim = lstm_hidden_dim
        self.num_layers = lstm_num_layers
        self.image_width = 224
        self.image_height = 224
        self.class_num = class_num
        if pretrained:
            self.load_state_dict(models.resnet50(pretrained=True).state_dict())
            #self.load_state_dict(models.resnet18(pretrained=False).state_dict())
            #self.load_state_dict(models.resnet101(pretrained=True).state_dict())

        _dropout = 0.3 #TODO:0.3
        cnn_out_size = 2048
        #cnn_out_size = 512 # for resnet18
        self.lstm = nn.LSTM(cnn_out_size, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        
        # linear
        self.hidden1_fc = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2_fc = nn.Linear(self.hidden_dim // 2, self.class_num)
        # dropout

    def forward(self, x):
        # size: batch, len, channel, width, height
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        #print(x.size())
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
        c_out = torch.flatten(cnn_x, 1) # batch*len, 2048/512
        lstm_in = c_out.view(batch_size, timesteps, 2048)

        lstm_out, _ = self.lstm(lstm_in)

        #print(lstm_out.shape)
        lstm_out = lstm_out[:,timesteps - 1,:]

        # linear
        cnn_lstm_out = self.hidden1_fc(torch.tanh(lstm_out))
        cnn_lstm_out = self.hidden2_fc(torch.tanh(cnn_lstm_out))
        # output
        logit = cnn_lstm_out
        #print(logit.shape)
        return logit       


    
def infer(model_in, file_name, fp16 = False, int8 = False):
    # === dataloader defination ===
    train_batch_size = 1
    valid_batch_size = 1
    test_batch_size = 1
    dataloaders = VSLdataset.create_dataloader_train_valid_test(train_batch_size, valid_batch_size, test_batch_size)

    valid_dataloader = dataloaders['valid']
    # =============================


    save_file = os.path.join('../saved_model', file_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model_in.eval().cuda()
    data_example = torch.randn(8, 10, 3, 224, 224)
    data_example = data_example.to(device)

    model_trt = torch2trt(model_in, [data_example], max_workspace_size=1<<26,fp16_mode = fp16, int8_mode = int8)
    model_trt.load_state_dict(torch.load(save_file))


    loss_function = nn.CrossEntropyLoss()

    # validation
    class_correct = list(0. for i in range(len(VSLdataset.class_name_to_id_)))
    class_total = list(0. for i in range(len(VSLdataset.class_name_to_id_)))
    class_name = list(VSLdataset.class_name_to_id_.keys())


    print('Test:')
    
    all_targets = np.zeros((len(valid_dataloader), 1))
    all_scores = np.zeros((len(valid_dataloader), 8))
    all_predicted_flatten = np.zeros((len(valid_dataloader), 1))

    loss_eval = 0.0
    for index_eval, (data_eval, target_eval) in enumerate(valid_dataloader):
        data_eval, target_eval = data_eval.to(device), target_eval.to(device)
        output_eval = model_trt(data_eval)

        loss_i = loss_function(output_eval, target_eval).item()
        loss_eval += loss_i

        all_targets[index_eval, :] = target_eval[0].cpu().detach().numpy()
        all_scores[index_eval, :] = output_eval[0].cpu().detach().numpy()
             
        _, predicted = torch.max(output_eval, 1)
        all_predicted_flatten[index_eval, :] = predicted[0].cpu().detach().numpy()
        if(predicted != target_eval): # batch_size, timesteps, C, H, W
            print('mis_classified: ', index_eval)
            #visualize_mis_class(data_eval[0].permute(0, 2, 3, 1).cpu(), str(index_eval) + '.png', class_name[target_eval[0].cpu().numpy()], class_name[predicted[0].cpu().numpy()])
        
        c = (predicted == target_eval).squeeze()
        for i in range(valid_batch_size):
            try:
                label = target_eval[i]
                class_correct[label] += c[i].item()
            except:
                label = target_eval
                class_correct[label] += c.item()
            class_total[label] += 1

    for i in range(len(VSLdataset.class_name_to_id_)):
        accuracy = 100 * (class_correct[i] + 1) / (class_total[i] + 1)
        print('Accuracy of %5s : %2d %%' % (
            class_name[i], accuracy))

    print('avg_loss: ', loss_eval/len(valid_dataloader))

    # === draw roc and confusion mat ===
    evaluate.draw_roc_bin(all_targets, all_scores)
    evaluate.draw_confusion_matrix(all_targets, all_predicted_flatten)
    # === draw roc and confusion mat end ===
            
def visualize_mis_class(frames, saved_name, true_label, false_label): # timesteps, C, H, W
    fig=plt.figure(figsize=(10, 6))
    fig.suptitle('GT: ' + true_label+'   Predicted:'+false_label)
    for i in range(10):
        fig.add_subplot(2,5,i+1) 
        plt.imshow(frames[i])
    save_file = os.path.join('../mis_classified', saved_name)
    plt.savefig(save_file)
    plt.close('all')

            
if __name__=='__main__':

    model = CLSTM(lstm_hidden_dim = 512, lstm_num_layers = 3, class_num=8)      
    infer(model, 'test_trt_int8.pth', fp16 = True, int8 = True)
