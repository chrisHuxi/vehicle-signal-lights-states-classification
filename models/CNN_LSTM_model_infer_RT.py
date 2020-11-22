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

from torch2trt import torch2trt

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

from torch2trt import TRTModule
import time


"""
    Neural Network: ResNet50_LSTM
    Detail: infer with original normal model and trt model
    
"""
# https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/4
# https://blog.csdn.net/shanglianlm/article/details/86376627 resnet usage
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

"""
    function to load the saved model
    @ param:
        1. model: the model object (must have the same architecture with the saved model)
        
        2. checkpoint_PATH: path of the saved model (eg. '../saved_model/CLSTM_50_l10_h512_d02.pth')
"""   
def load_checkpoint(model, checkpoint_PATH):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['model_state_dict'])
        print('loading checkpoint!')
    return model   


"""
    function to save the pytorch model into trt model
    @ param:
        1. model_in: the model object (must have the same architecture with the saved model)
        
        2. file_name: saved model name (eg. test_trt_fp32.pth)
        
        3. fp16: enable FP16 mode, which is not supported by gtx1060
        
        4. int8: enable INT8 mode, which is not clear how to use sofar
"""   
def save_model_trt(model_in, file_name, fp16 = False, int8 = False):
    # === got model ===
    save_file = os.path.join('../saved_model', 'CLSTM_50_l10_h512_loss021_best.pth')
    model = load_checkpoint(model_in, save_file)
    # =================
    print(model)

    # === runing no gpu ===
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    torch.backends.cudnn.benchmark = True
    model.eval()
    # =====================
    data_example = torch.ones((8, 10, 3, 224, 224))
    data_example = data_example.to(device)


    model_trt = torch2trt(model, [data_example], max_workspace_size=1<<26, fp16_mode = fp16, int8_mode = int8)
    torch.save(model_trt.state_dict(), file_name)

"""
    function to infer with the pytorch model
    @ param:
        1. model_in: the model object (must have the same architecture with the saved model)
"""  
def infer_normal(model_in):

    # === got model ===
    save_file = os.path.join('../saved_model', 'CLSTM_50_l10_h512_loss021_best.pth')
    model = load_checkpoint(model_in, save_file)
    # =================
    print(model)

    # === runing no gpu ===
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    torch.backends.cudnn.benchmark = True
    model.eval()
    # =====================
    data_example = torch.ones((1, 10, 3, 224, 224))
    data_example = data_example.to(device)


    for folder_name in os.listdir('/media/huxi/DATA/inf_master/Semester-5/Thesis/code/dataset/inference_data'):
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
        mean_runtime = 0
        try:
            for index_eval, (data_eval, frame_id_eval) in enumerate(infer_dataloader):
                data_eval = data_eval.to(device)

                torch.cuda.current_stream().synchronize()
                start_time = time.time()
                output_eval = model(data_eval)
                torch.cuda.current_stream().synchronize()

                end_time = time.time()
                print("--- %s seconds --- normal" % (time.time() - start_time))

                mean_runtime += end_time - start_time

                print(torch.cuda.memory_cached(0)/(1024.0*1024))
                print(torch.cuda.memory_allocated(0)/(1024.0*1024))


                _, predicted = torch.max(output_eval, 1)
                print_list.append([str(frame_id_eval[0]), class_name[predicted[0]]])
        except:
            pass
        print("mean runtime:")
        print(mean_runtime/index_eval)
        break
 
    print('test')



"""
    function to load and infer with trt model
    @ param:
        1. model_in: the model object (must have the same architecture with the saved model)
        
        2. file_name: saved model name (eg. test_trt_fp32.pth)
        
        3. fp16: enable FP16 mode, which is not supported by gtx1060
        
        4. int8: enable INT8 mode, which is not clear how to use sofar
""" 
def load_trt_model_and_infer(model_in, file_name, fp16 = False, int8 = False):

    save_file = os.path.join('../saved_model', file_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model_in.eval().cuda()
    data_example = torch.randn(1, 10, 3, 224, 224)
    data_example = data_example.to(device)

    model_trt = torch2trt(model_in, [data_example], max_workspace_size=1<<26,fp16_mode = fp16, int8_mode = int8)
    model_trt.load_state_dict(torch.load(save_file))


    for folder_name in os.listdir('/media/huxi/DATA/inf_master/Semester-5/Thesis/code/dataset/inference_data'):
        print(folder_name)
        track_folder_name = folder_name

        # === dataloader defination ===
        dataloaders = VSLdataset.create_dataloader_infer(1, track_folder_name)
        class_name = list(VSLdataset.class_name_to_id_.keys())

        infer_dataloader = dataloaders['infer']
        # =============================
        print_list = []
        result_file_name = track_folder_name + '.txt'
        mean_runtime = 0
        print('Infer:')

        mean_runtime = 0
        try:
            for index_eval, (data_eval, frame_id_eval) in enumerate(infer_dataloader):
                data_eval = data_eval.to(device)
                # benchmark tensorrt throughput
                torch.cuda.current_stream().synchronize()
                start_time = time.time()
                output_eval_trt = model_trt(data_eval)
                torch.cuda.current_stream().synchronize()
                end_time = time.time()
                print("--- %s seconds --- trt" % (end_time - start_time))
                mean_runtime += end_time - start_time

                print(torch.cuda.memory_cached(0)/(1024.0*1024))
                print(torch.cuda.memory_allocated(0)/(1024.0*1024))


                _, predicted_trt = torch.max(output_eval_trt, 1)
                print_list.append([str(frame_id_eval[0]), class_name[predicted_trt[0]]])
        except:
            pass
        print("mean runtime:")
        print(mean_runtime/index_eval)
        break

    print('test')


"""
    function to test a original resnet50's gpu memory cost
    @ param:
        None
""" 
def resnet_test():

    # === got model ===
    model = models.resnet.resnet50(pretrained=True)
    # =================
    print(model)

    # === runing no gpu ===
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.backends.cudnn.benchmark = True
    model.eval()
    # =====================
    for i in range(1000):
        x = torch.ones((10, 3, 224, 224))
        output_eval = model(x.cuda())
        print(output_eval)


"""
    a example to construct the model and infer with it by trt
"""   
if __name__=='__main__':

    model = CLSTM(lstm_hidden_dim = 512, lstm_num_layers = 3, class_num=8) 
    #save_model_trt(model, 'test_trt_int8.pth', fp16 = False, int8 = True)     
    #infer_normal(model)
    #resnet_test()
    load_trt_model_and_infer(model, 'test_trt_fp32.pth', fp16 = True, int8 = False)



