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
import dataloader.VSLdataset_mid as VSLdataset
#import dataloader.VSLdataset_short as VSLdataset
#import dataloader.VSLdataset as VSLdataset

import torch.optim as optim
    
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter    

import evaluate

"""
    Neural Network: CNN_LSTM
    Detail: the input crosss cnn model and LSTM model independly, then the result of both concat
"""

from torch.autograd import Variable


# https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/4
# https://blog.csdn.net/shanglianlm/article/details/86376627 resnet 用法
class CLSTM(models.resnet.ResNet):
    def __init__(self, lstm_hidden_dim, lstm_num_layers, class_num, pretrained=True):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3]) # 50

        self.hidden_dim = lstm_hidden_dim
        self.num_layers = lstm_num_layers
        self.image_width = 224
        self.image_height = 224
        self.class_num = class_num
        if pretrained:
            self.load_state_dict(models.resnet50(pretrained=True).state_dict())


        _dropout = 0.2 #TODO:0.3
        cnn_out_size = 2048
        #cnn_out_size = 512 # for resnet18
        self.lstm = nn.LSTM(cnn_out_size, self.hidden_dim, dropout=_dropout, num_layers=self.num_layers, batch_first=True)
        
        # linear
        self.hidden1_fc = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2_fc = nn.Linear(self.hidden_dim // 2, self.class_num)
        # dropout

        self.dropout_cnn0 = nn.Dropout(p=_dropout)
        self.dropout_cnn1 = nn.Dropout(p=_dropout)
        self.dropout_cnn2 = nn.Dropout(p=_dropout)

        self.dropout_fc = nn.Dropout(p=_dropout)


    # https://github.com/HHTseng/video-classification/blob/master/CRNN/functions.py    
    def forward(self, x):
        # size: batch, len, channel, width, height
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)

        # ResNet:
        cnn_x = self.conv1(c_in)

        cnn_x = self.bn1(cnn_x)
        cnn_x = self.relu(cnn_x)
        cnn_x = self.maxpool(cnn_x)
        cnn_conv_f1 = cnn_x

        cnn_x = self.layer1(cnn_x)

        cnn_x = self.dropout_cnn0(cnn_x)
        cnn_x = self.layer2(cnn_x)
        cnn_conv_f2 = cnn_x
        cnn_x = self.dropout_cnn1(cnn_x)
        cnn_x = self.layer3(cnn_x)
        cnn_x = self.dropout_cnn2(cnn_x)
        cnn_x = self.layer4(cnn_x)

        
        cnn_x = self.avgpool(cnn_x)
        c_out = torch.flatten(cnn_x, 1) # batch*len, 2048/512
        lstm_in = c_out.view(batch_size, timesteps, -1)
        
        lstm_out, _ = self.lstm(lstm_in)
        #print(lstm_out.shape)

        # linear
        cnn_lstm_out = self.hidden1_fc(torch.tanh(lstm_out[:,-1,:]))
        cnn_lstm_out = self.dropout_fc(cnn_lstm_out) # shall we add the dropout in fc?
        cnn_lstm_out = self.hidden2_fc(torch.tanh(cnn_lstm_out))
        # output
        logit = cnn_lstm_out
        #print(logit.shape)
        return logit, cnn_conv_f1       


def load_checkpoint(model, checkpoint_PATH):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['model_state_dict'])
        print('loading checkpoint!')
    return model   


from torchvision.utils import make_grid
from PIL import Image
# https://blog.csdn.net/whtlook/article/details/98228855
def save_img(tensor, name):
    tensor = tensor.permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=10, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')

def infer(model_in):
    # === dataloader defination ===
    train_batch_size = 1
    valid_batch_size = 1
    test_batch_size = 1
    #dataloaders = VSLdataset.create_dataloader_train_valid_test(train_batch_size, valid_batch_size, test_batch_size)
    dataloaders = VSLdataset.create_dataloader_valid(valid_batch_size)

    valid_dataloader = dataloaders['valid']
    # =============================

    # === got model ===
    save_file = os.path.join('../saved_model', 'CLSTM_50_l10_h512_loss021_best.pth')
    model = load_checkpoint(model_in, save_file)
    # =================
    print(model)

    # === runing no gpu ===
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.backends.cudnn.benchmark = True
    # =====================

    loss_function = nn.CrossEntropyLoss()

    # validation
    class_correct = list(0. for i in range(len(VSLdataset.class_name_to_id_)))
    class_total = list(0. for i in range(len(VSLdataset.class_name_to_id_)))
    class_name = list(VSLdataset.class_name_to_id_.keys())
    model.eval()
    print('Test:')
    
    all_targets = np.zeros((len(valid_dataloader), 1))
    all_scores = np.zeros((len(valid_dataloader), 8))
    all_predicted_flatten = np.zeros((len(valid_dataloader), 1))

    loss_eval = 0.0
    for index_eval, (data_eval, target_eval) in enumerate(valid_dataloader):
        data_eval, target_eval = data_eval.to(device), target_eval.to(device)
        output_eval, cnn_conv_f1 = model(data_eval)
        
        loss_i = loss_function(output_eval, target_eval).item()
        loss_eval += loss_i

        all_targets[index_eval, :] = target_eval[0].cpu().detach().numpy()
        all_scores[index_eval, :] = output_eval[0].cpu().detach().numpy()
             
        _, predicted = torch.max(output_eval, 1)
        all_predicted_flatten[index_eval, :] = predicted[0].cpu().detach().numpy()

        print('label:')
        print(class_name[target_eval[0]])

        print('predicted:')
        print(class_name[predicted[0]])

        
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

    evaluate.calculate_f1(all_targets, all_predicted_flatten)
            
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
    infer(model)
