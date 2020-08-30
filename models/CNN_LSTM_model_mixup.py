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
#import dataloader.VSLdataset_yoloraw as VSLdataset
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
    

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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
    save_file = os.path.join('../saved_model', 'CLSTM_50_l10_h512_mixup.pth')
    writer = SummaryWriter('../saved_model/tensorboard_log_50_l10_h512_mixup')
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
    #alpha = torch.tensor([1.0, 1.0, 1.0, 1.05, 1.2, 1.0, 1.0, 1.0]) #试试1.2?
    #loss_function = FocalLoss(class_num = 8, alpha=alpha, gamma=0, size_average=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=1,patience=2)

    # === runing no gpu ===
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.backends.cudnn.benchmark = True
    # =====================
    # epoch = 1

    for epoch in range(num_epochs):
        # training
        model.train()
        train_loss  = 0.0
        print('Train:')

        for index, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            inputs, targets_a, targets_b, lam = mixup_data(data, target, 1.0)
            inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
            outputs = model(inputs)
            loss = mixup_criterion(loss_function, outputs, targets_a, targets_b, lam)

            
            #output = model(data)
            #loss = loss_function(output, target)
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
    save_file = os.path.join('../saved_model', 'CLSTM_50_l10_h512_focal.pth')
    model = load_checkpoint(model_in, save_file)
    # =================
    print(model)

    # === runing no gpu ===
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.backends.cudnn.benchmark = True
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
    model = CLSTM(lstm_hidden_dim = 512, lstm_num_layers = 3, class_num=8)        
    train(model_in = model, num_epochs = 100, load_model = True, freeze_extractor = False)

    #model = CLSTM(lstm_hidden_dim = 512, lstm_num_layers = 4, class_num=8)      
    #infer(model)
