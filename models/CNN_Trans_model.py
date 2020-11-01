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
import dataloader.VSLdataset_long as VSLdataset
#import dataloader.VSLdataset as VSLdataset

import torch.optim as optim
    
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter    

# transformer encoder:
'''
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(10, 32, 512)
out = transformer_encoder(src)
print(out.shape) #(10, 32, 512)
'''

# https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/selfAttention.py
# 明天看看吧
class SelfAttention(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
		super(SelfAttention, self).__init__()

		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		--------
		
		"""

		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		self.weights = weights

		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
		self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
		self.dropout = 0.8
		self.bilstm = nn.LSTM(embedding_length, hidden_size, dropout=self.dropout, bidirectional=True)
		# We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
		self.W_s1 = nn.Linear(2*hidden_size, 350)
		self.W_s2 = nn.Linear(350, 30)
		self.fc_layer = nn.Linear(30*2*hidden_size, 2000)
		self.label = nn.Linear(2000, output_size)

	def attention_net(self, lstm_output):

		"""
		Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
		encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of 
		the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully 
		connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e., 
		pos & neg.
		Arguments
		---------
		lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
		---------
		Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
				  attention to different parts of the input sentence.
		Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
					  attn_weight_matrix.size() = (batch_size, 30, num_seq)
		"""
		attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
		attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
		attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

		return attn_weight_matrix

	def forward(self, input_sentences, batch_size=None):

		""" 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for pos & neg class.
		
		"""

		input = self.word_embeddings(input_sentences)
		input = input.permute(1, 0, 2)
		if batch_size is None:
			h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
		else:
			h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())

		output, (h_n, c_n) = self.bilstm(input, (h_0, c_0))
		output = output.permute(1, 0, 2)
		# output.size() = (batch_size, num_seq, 2*hidden_size)
		# h_n.size() = (1, batch_size, hidden_size)
		# c_n.size() = (1, batch_size, hidden_size)
		attn_weight_matrix = self.attention_net(output)
		# attn_weight_matrix.size() = (batch_size, r, num_seq)
		# output.size() = (batch_size, num_seq, 2*hidden_size)
		hidden_matrix = torch.bmm(attn_weight_matrix, output)
		# hidden_matrix.size() = (batch_size, r, 2*hidden_size)
		# Let's now concatenate the hidden_matrix and connect it to the fully connected layer.
		fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
		logits = self.label(fc_out)
		# logits.size() = (batch_size, output_size)

		return logits
# https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/4
# https://blog.csdn.net/shanglianlm/article/details/86376627 resnet 用法
# https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
class CTrans(models.resnet.ResNet):
    def __init__(self, trans_num_layers, class_num, pretrained=True):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3]) # 50

        self.num_layers = trans_num_layers
        self.image_width = 224
        self.image_height = 224
        self.class_num = class_num
        if pretrained:
            self.load_state_dict(models.resnet50(pretrained=True).state_dict())
            #self.load_state_dict(models.resnet18(pretrained=False).state_dict())
            #self.load_state_dict(models.resnet101(pretrained=True).state_dict())

        _dropout = 0.3
        cnn_out_size = 2048
        #cnn_out_size = 512 # for resnet18

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=cnn_out_size, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.num_layers)

        # linear
        self.hidden1_fc = nn.Linear(cnn_out_size, cnn_out_size // 2)
        self.hidden2_fc = nn.Linear(cnn_out_size // 2, self.class_num)
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
        transformer_in = c_out.view(batch_size, timesteps, -1)
        
        transformer_out = self.transformer_encoder(transformer_in)

        transformer_out = torch.sum(transformer_out, 1)
        #print(transformer_out.shape)

        # linear
        cnn_transformer_out = self.hidden1_fc(torch.tanh(transformer_out))
        cnn_transformer_out = self.dropout_fc(cnn_transformer_out) # shall we add the dropout in fc?
        cnn_transformer_out = self.hidden2_fc(torch.tanh(cnn_transformer_out))
        # output
        logit = cnn_transformer_out
        #print(logit.shape)
        return logit       


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
    save_file = os.path.join('../saved_model', 'CTrans_50_l10_h512_d01.pth')
    writer = SummaryWriter('../saved_model/tensorboard_log_CTrans_50_l10_h512_d01')
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
        output_eval = model(data_eval)

        loss_i = loss_function(output_eval, target_eval).item()
        loss_eval += loss_i

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
    #test_model()
    model = CTrans(trans_num_layers = 6, class_num=8)        
    train(model_in = model, num_epochs = 100, load_model = False, freeze_extractor = False)

    #model = CLSTM(lstm_hidden_dim = 512, lstm_num_layers = 3, class_num=8)      
    #infer(model)
