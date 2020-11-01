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
import dataloader.VSLdataset_infer as VSLdataset

import torch.optim as optim
    
import glob

# https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/4
# https://blog.csdn.net/shanglianlm/article/details/86376627 resnet 用法
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

        cnn_x = self.layer1(cnn_x)
        cnn_x = self.dropout_cnn0(cnn_x)
        cnn_x = self.layer2(cnn_x)
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
        return logit       


def load_checkpoint(model, checkpoint_PATH):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['model_state_dict'])
        print('loading checkpoint!')
    return model   

def infer(model_in):

    # === got model ===
    save_file = os.path.join('../saved_model', 'CLSTM_50_l10_h512_loss021_best.pth')
    model = load_checkpoint(model_in, save_file)
    # =================
    print(model)
    import time

    # === runing no gpu ===
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.backends.cudnn.benchmark = True
    model.eval()
    # =====================
    for folder_name in os.listdir('/media/huxi/DATA/inf_master/Semester-5/Thesis/code/dataset/inference_data'):
        try:

            print(folder_name)
            track_folder_name = folder_name


            # === dataloader defination ===
            dataloaders = VSLdataset.create_dataloader_infer(1, track_folder_name)
            class_name = list(VSLdataset.class_name_to_id_.keys())

            infer_dataloader = dataloaders['infer']
            # =============================
            print_list = []
            file_name = track_folder_name + '.txt'
            print('Infer:')
        
            for index_eval, (data_eval, frame_id_eval) in enumerate(infer_dataloader):
                print(index_eval)
                data_eval = data_eval.to(device)
                start_time = time.time()
                output_eval = model(data_eval)
                print("--- %s seconds ---" % (time.time() - start_time))

                _, predicted = torch.max(output_eval, 1)
                print(frame_id_eval[0])
                print(class_name[predicted[0]])

                score = output_eval[0]
                sf_fun = nn.Softmax()
                prob = sf_fun(score)

                prob = prob.cpu().detach().numpy()
                print(prob)
                print_list.append([str(frame_id_eval[0]), class_name[predicted[0]], prob])
        except:
            pass

        fileObject = open(file_name, 'w')
        for line in print_list:
            fileObject.write(line[0])
            fileObject.write(" ")
            fileObject.write(line[1])
            fileObject.write(" ")
            fileObject.write(" ".join(map(str, line[2])))
            fileObject.write('\n')
        fileObject.close()
        
            
if __name__=='__main__':

    model = CLSTM(lstm_hidden_dim = 512, lstm_num_layers = 3, class_num=8)      
    infer(model)
