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
#import dataloader.VSLdataset as VSLdataset
import dataloader.VSLdataset_mask as VSLdataset
import torch.optim as optim
    
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter    

import evaluate

"""
self defined adaptive weigthed mask layer 
"""
class WeightedMaskLayer(nn.Module):
    def __init__(self, channel_num, width, height):
        super(WeightedMaskLayer, self).__init__()
        self.mask_weight = torch.nn.Parameter(data=torch.Tensor(channel_num, width, height), requires_grad=True)
        self.mask_weight.data.uniform_(-1, 1)
    
    def forward(self, c_in_data, c_in_mask):

        expand_mask_weight = self.mask_weight.expand(c_in_mask.shape[0], c_in_mask.shape[1], c_in_mask.shape[2], c_in_mask.shape[3])
        expand_mask_weight = torch.sigmoid(expand_mask_weight)

        c_in_both = c_in_data * torch.pow(c_in_mask, expand_mask_weight)
        return c_in_both


"""
    Neural Network: CNN_LSTM
    Detail: the input crosss cnn model and LSTM model independly, then the result of both concat
"""
# https://discuss.pytorch.org/t/attention-in-image-classification/80147/3 self-attention
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
        
        self.weighted_mask_layer = WeightedMaskLayer(2048, 7, 7)

        self.conv_1_mask = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,stride=2,padding=3, bias=False), # padding: both in head and tail ==> padded result
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        #out size: 112

        self.conv_2_mask = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1, bias=False), 
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        #out size: 56

        self.conv_3_mask = nn.Sequential(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1, bias=False), 
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        #out size: 28

        self.conv_4_mask = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1, bias=False), 
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        #out size: 14


        self.conv_5_mask = nn.Sequential(nn.Conv2d(in_channels=512,out_channels=2048,kernel_size=3,stride=1,padding=1, bias=False), 
                                       nn.BatchNorm2d(2048),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        #out size: 7


        '''
        self.conv_for_concat = nn.Sequential(
                                       nn.Conv2d(in_channels=72,out_channels=64,kernel_size=7,stride=1,padding=3, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True))#,
                                       #nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        '''
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
    def forward(self, x_xmask): # mask
        # TODO: add mask into network
        x, x_mask = x_xmask
        

        # size: batch, len, channel, width, height
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_in_mask = x_mask.view(batch_size * timesteps, 1, H, W)

        #c_in = torch.cat([c_in, c_mask_in], dim = 1)


        # ResNet:
        c_in_data = self.conv1(c_in)
        cnn_x = c_in_data
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

        c_in_mask = F.normalize(c_in_mask.view(c_in_mask.size(0), -1), dim=1, p=2).view(c_in_mask.size())
        c_in_mask = self.conv_1_mask(c_in_mask)
        c_in_mask = self.conv_2_mask(c_in_mask)
        c_in_mask = self.conv_3_mask(c_in_mask)
        c_in_mask = self.conv_4_mask(c_in_mask)
        c_in_mask = self.conv_5_mask(c_in_mask)

        c_in_both = self.weighted_mask_layer(cnn_x, c_in_mask)

        cnn_x = self.avgpool(c_in_both)
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
    
def train(model_in, num_epochs = 3, load_model = True, freeze_extractor = True):
    # === dataloader defination ===
    train_batch_size = 2
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

    # === got model ===
    save_file = os.path.join('../saved_model', 'CLSTM_50_l10_h512_yolo_gaussmask.pth')
    writer = SummaryWriter('../saved_model/tensorboard_log_50_l10_h512_yolo_gaussmask')
    if(load_model == True):
        model = load_checkpoint(model_in, save_file)
    else:
        model = model_in
    # =================
    print(model)
    # === freeze some layers against overfitting ===
    if(freeze_extractor == True):
        for layer_id, child in enumerate(model.children()):
            if layer_id < 8: #layer 9 is the fc
                for param in child.parameters():
                    param.requires_grad = False
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.0001) #lr=0.0000000001)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # ==============================================

    loss_function = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=1,patience=2)

    # === runing no gpu ===
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.backends.cudnn.benchmark = False
    # =====================
    # epoch = 1

    for epoch in range(num_epochs):
        # training
        model.train()
        train_loss  = 0.0
        print('Train:')
        
        for index, ((data, mask), target) in enumerate(train_dataloader): # mask
            data, mask, target = data.to(device), mask.to(device), target.to(device)
            optimizer.zero_grad()
            output = model((data, mask))
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if index % 10  == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, index + 1, train_loss / 10))
                writer.add_scalar('Train/Loss', train_loss / 10, epoch * (len(train_dataloader)) + index + 1)
                #writer.add_image('weight_mask', mask_weight, epoch * (len(train_dataloader)) + index + 1)
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
            for index_eval, ((data_eval, mask_eval), target_eval) in enumerate(valid_dataloader):
                data_eval, mask_eval, target_eval = data_eval.to(device), mask_eval.to(device), target_eval.to(device)
                output_eval = model((data_eval, mask_eval))
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
                    loss_for_display = 0.0

            for i in range(len(VSLdataset.class_name_to_id_)):
                accuracy = 100 * (class_correct[i] + 1) / (class_total[i] + 1)
                print('Accuracy of %5s : %2d %%' % (
                    class_name[i], accuracy))
                # Record loss and accuracy from the test run into the writer
                writer.add_scalar('Valid/Accuracy ' + str(class_name[i]), accuracy, epoch)
                writer.flush()
            print('avg_loss: ', loss_eval/len(valid_dataloader))
            scheduler.step(loss_eval/len(valid_dataloader))
            writer.add_scalar('Valid/Loss ', loss_eval/len(valid_dataloader), epoch)
            writer.flush()
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

def infer(model_in):
    # === dataloader defination ===
    train_batch_size = 1
    valid_batch_size = 1
    test_batch_size = 1
    dataloaders = VSLdataset.create_dataloader_train_valid_test(train_batch_size, valid_batch_size, test_batch_size)
    valid_dataloader = dataloaders['valid']
    # =============================

    # === got model ===
    save_file = os.path.join('../saved_model', 'CLSTM_50_l10_h512_yolocut.pth')
    model = load_checkpoint(model_in, save_file)
    # =================
    print(model)

    # === runing no gpu ===
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.backends.cudnn.benchmark = False
    # =====================

    # validation
    class_correct = list(0. for i in range(len(VSLdataset.class_name_to_id_)))
    class_total = list(0. for i in range(len(VSLdataset.class_name_to_id_)))
    class_name = list(VSLdataset.class_name_to_id_.keys())
    model.eval()
    print('Test:')
    
    all_targets = np.zeros((len(valid_dataloader), 1))
    all_scores = np.zeros((len(valid_dataloader), 8))
    all_predicted_flatten = np.zeros((len(valid_dataloader), 1))
    
    for index_eval, (data_eval, target_eval) in enumerate(valid_dataloader):
        data_eval, target_eval = data_eval.to(device), target_eval.to(device)
        output_eval = model(data_eval)
        
        all_targets[index_eval, :] = target_eval[0].cpu().detach().numpy()
        all_scores[index_eval, :] = output_eval[0].cpu().detach().numpy()
             
        _, predicted = torch.max(output_eval, 1)
        all_predicted_flatten[index_eval, :] = predicted[0].cpu().detach().numpy()
        if(predicted != target_eval): # batch_size, timesteps, C, H, W
            print('mis_classified: ', index_eval)
            visualize_mis_class(data_eval[0].permute(0, 2, 3, 1).cpu(), str(index_eval) + '.png', class_name[target_eval[0].cpu().numpy()], class_name[predicted[0].cpu().numpy()])
        
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
    #test_model()
    #model = CLSTM(lstm_hidden_dim = 512, lstm_num_layers = 4, class_num=8)        
    #train(model_in = model, num_epochs = 100, load_model = False, freeze_extractor = False)

    model = CLSTM(lstm_hidden_dim = 512, lstm_num_layers = 4, class_num=8)      
    infer(model)
