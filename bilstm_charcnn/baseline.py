from __future__ import print_function
import time
start = time.time()
from collections import Counter, defaultdict
import random
#import sys
import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
#import emot
import torch.nn.init

from torch.nn import init

import re
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


import sys
sys.stdout = open('loss.log', 'w', 0)

WEMBED_SIZE = 200
CEMBED_SIZE = 50
HIDDEN_SIZE = 200
MLP_SIZE = 200
TIMEOUT = 300000
START_TAG= '<start>'
STOP_TAG= '<stop>'

#def compute_accuracy():
    
#def compute_precision():

#def compute_recall():

#def compute_F1():


def init_xavier(m):
    if isinstance(m, nn.LSTM):
        nn.init.xavier_normal(m.weight_hh_l0)
        nn.init.xavier_normal(m.weight_ih_l0)


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec): #need
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class Model(nn.Module):

    def __init__(self, nwords, ntags, UNK):
        super(Model, self).__init__()
        self.tag_to_ix = np.load("tags_new.npy").item()
        self.tagset_size = len(self.tag_to_ix)
        self.lookup_w = nn.Embedding(nwords, WEMBED_SIZE, padding_idx=UNK)
        #self.conv13 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_SIZE, kernel_size=(CEMBED_SIZE, 3))
        #self.conv14 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_SIZE, kernel_size=(CEMBED_SIZE, 4))
        #self.conv15 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_SIZE, kernel_size=(CEMBED_SIZE, 5))
        #self.lstm = nn.LSTM(3 * HIDDEN_SIZE, HIDDEN_SIZE, 3 ,  batch_first = True, bidirectional = True)
        self.lstm = nn.LSTM(WEMBED_SIZE, HIDDEN_SIZE, 1, bidirectional = True)
        #self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        #self.proj1 = nn.Linear(2 * HIDDEN_SIZE, self.tagset_size)
        self.proj1 = nn.Linear(2 * HIDDEN_SIZE, 6)
        #self.proj1.weight = self.lookup_w.weight
        #self.proj2 = nn.Linear(MLP_SIZE, self.tagset_size)
        #self.proj1 = nn.Linear(2 * HIDDEN_SIZE, MLP_SIZE)
        #self.proj2 = nn.Linear(MLP_SIZE, ntags)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.lookup_w.weight.data.uniform_(-initrange, initrange)
        self.proj1.bias.data.fill_(0)
        #self.proj2.bias.data.fill_(0)
        #self.proj1.weight.data.uniform_(-initrange, initrange)
        #self.proj2.weight.data.uniform_(-initrange, initrange)
    
    def init_hidden(self):
        if torch.cuda.is_available():
            return (Variable(torch.randn(2, 1, HIDDEN_SIZE)).cuda(),
                    Variable(torch.randn(2, 1, HIDDEN_SIZE)).cuda())
        else:
            return (Variable(torch.randn(2, 1, HIDDEN_SIZE)),
                    Variable(torch.randn(2, 1, HIDDEN_SIZE)))
    
        
       
    def forward(self, words, seq_lengths):
        embeddings = self.lookup_w(words)
        embeddings = pack_padded_sequence(embeddings, seq_lengths) #packed_input
        #x2 = self.conv13(embeddings)
        #x2 = F.relu(x2)
        #x2 = F.max_pool1d(x2, x2.size(2))
        #x3 = self.conv13(embeddings)
        #x3 = F.relu(x3)
        #x3 = F.max_pool1d(x3, x3.size(2))
        #x4 = self.conv13(embeddings)
        #x4 = F.relu(x4)
        #x4 = F.max_pool1d(x4, x4.size(2))
        #embeddings = torch.cat((x2, x3, x4), 1)
        embeddings, (hidden, state) = self.lstm(embeddings)
        #embeddings = self.dropout(embeddings)
        #embeddings = torch.transpose(embeddings,2,1)
        embeddings, _ = pad_packed_sequence(embeddings)

        #embeddings = F.max_pool1d(embeddings, embeddings.size(2))

        embeddings = self.dropout(hidden)
        embeddings=embeddings.view(embeddings.size(1), -1)

        embeddings = self.proj1(embeddings)
        #F.sigmoid(embeddings)
        #embeddings = self.proj2(embeddings)
        #embeddings = self.tanh(embeddings)
        #embeddings = self.proj2(embeddings)
        return embeddings,hidden


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self,data, labels):
        self.data = data
        self.labels = labels
    def __getitem__(self, index):
        sent = self.data[index]
        label = self.labels[index]
        return torch.from_numpy(sent), torch.from_numpy(label) 
        

    def __len__(self):
        return len(self.data)


# would be used for convolution..not using currently
class CrossEntropyLoss3D(nn.CrossEntropyLoss):
    def forward(self, input, target, reduce= False):
        return super(CrossEntropyLoss3D, self).forward(input.view(-1, input.size()[2]), target.view(-1), reduce = False)


def custom_collate(batch): 
    
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    data, labels = zip(*batch)
    #list containing the lengths of each sentence
    seq_len = [len(d) for d in data]
    #get the max length of all the sentences
    max_len = max(seq_len)

    targets = torch.zeros(max(seq_len), len(data))
    
    mask= torch.zeros(len(labels),max(seq_len))
    
    label = torch.zeros(1,len(labels))
    for i in range(len(data)):
        mask[:seq_len[i],i] = 1
        
    for i, d in enumerate(data):
        end = seq_len[i]
        targets[:end, i] = d[:end]

    #print("targets is: ",targets)
    #print("labels is: ",torch.stack(labels))
    label=torch.stack(labels)
    return targets, torch.transpose(label,0,1), np.array(seq_len), mask


# loading numpy arrays for sentences and labels
train_file_load = np.load("train_words.npy")
train_labels_load = np.load("train_labels.npy")
idx = np.random.permutation(train_file_load.shape[0])
x,y = train_file_load[idx], train_labels_load[idx]
train_file = train_file_load[:int(0.8*x.shape[0])]
train_labels = train_labels_load[:int(0.8*y.shape[0])]
print(x.shape,y.shape)
dev_file = train_file_load[int(0.8*x.shape[0]):]
dev_labels = train_labels_load[int(0.8*y.shape[0]):]



#dev_file = np.load("~/cmner/NER/dev_words.npy")
#dev_labels = np.load("~/cmner/NER/dev_labels.npy")
#test_file = "~/cmner/NER/test.txt"
words = np.load("vocab.npy")


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] =  param_group['lr']  / ( 1 + epoch * np.sqrt(2))

probsf = open("probs.txt", 'w')

def inference(model, loader): 
    model.eval() 
    count = 0 
    epoch_loss = 0 
    loss_fn = nn.MultiLabelSoftMarginLoss()
    for batch_idx, (data, label, seq_len, mask) in enumerate(loader):
        count = count + 90
        
        '''
        if torch.cuda.is_available:
            print("WHY AM I HERE")
            data = data.cuda()
            label = label.cuda()
            mask = mask.cuda()
        '''     
        X = Variable(data).long()
        Y = Variable(label).float()
        
        mask = Variable(mask)
        indices = torch.nonzero(mask.view(-1))
        
        out,ht = model.forward(X, seq_len)
        loss = loss_fn(out,Y.transpose(0,1))
        out=F.sigmoid(out)
        probsf.write("----")
        probsf.write(str(out.data.numpy().tolist()))
        t = Variable(torch.Tensor([0.5]))
        out = (out > t).float() * 1
        print(str(Y.transpose(0,1).data.numpy().tolist())+" | "+str(out.data.numpy().tolist()))
        epoch_loss += loss[0].data.cpu()
    return epoch_loss / count

def test_function(model, loader, tags):
    model.eval()
    fout = open("stress.pred","w")
    calcs_test = dev_file
    
    for batch_idx, (data, label, seq_len, mask) in enumerate(loader):
        
        tag = []
        if torch.cuda.is_available():
            data = data.cuda()
            
                    
        X = Variable(data).long()    
        out,ht =model.forward(X, seq_len)
        #print(out)
        seq_len = seq_len[0]
        #preds = out.max(2)[1]
        preds=out
        print(F.sigmoid(preds),X.size())
        for i in range(preds.size(0)):
            pred = preds[i,0].data.cpu().numpy()[0]
            fout.write(calcs_test[batch_idx][i] +"\t" + str(tags[pred]).upper() + "\n" )
        fout.write("\n")
    fout.close()
        
        
    #return tag
    
    
class Trainer(): 
    """ A simple training cradle """
    def __init__(self, model, optimizer, batch_size = 64, load_path=None):
        self.model = model
        self.loss_fn = nn.MultiLabelSoftMarginLoss()
        #self.loss_fn = nn.CrossEntropyLos()
        if load_path is not None:
            self.model = torch.load(load_path)
        self.optimizer = optimizer
        self.batch_size = batch_size
        
    def stop_cond(self):
        # TODO: Implement early stopping
        def deriv(ns):
            return [ns[i+1] - ns[i] for i in range(len(ns)-1)]
        val_errors = [m.val_error for m in self.metrics]
        back = val_errors[-10:]
        return sum(deriv(back)) > 0
            
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def run(self, epochs):
        print ("nwords=%r, ntags=%r " % (nwords, ntags))
        print("begin training...")

        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()
        
        
        train_dataset = MyDataSet(train_file, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 90, collate_fn = custom_collate, shuffle = True, num_workers=1, pin_memory=True)

        
        dev_dataset = MyDataSet(dev_file, dev_labels)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size = 1, collate_fn = custom_collate, shuffle = True, num_workers=1, pin_memory=True)

        self.metrics = []
        for e in range(n_epochs):
            model.train()
            losses = []
            if self.stop_cond():
                return
            epoch_loss = 0
            count = 0
            torch.manual_seed(3000)
            for batch_idx, (data, label, seq_len, mask) in enumerate(train_loader):
                print(data.size(),label.size())
                count = count + 90
                self.optimizer.zero_grad()
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()
                    mask = mask.cuda()
                X = Variable(data).long()
                Y = Variable(label).float()
                mask = Variable(mask)
                indices = torch.nonzero(mask.view(-1))
                
                out,ht = self.model(X, seq_len)
                #loss = self.loss_fn(torch.index_select(out.view(-1, out.size()[2]), 0, indices.squeeze(1)), torch.index_select(Y.view(-1),0,indices.squeeze(1)))
                loss = self.loss_fn(out,Y.transpose(0,1))
                loss.backward()
                nn.utils.clip_grad_norm(self.model.parameters(), 0.25)
                self.optimizer.step()
                epoch_loss += loss[0].data.cpu()
                
                if batch_idx % 100  == 0:
                    print(loss[0].data.cpu().numpy()[0])
                #tensor_logger.model_param_histo_summary(model, (e * 32) + batch_idx)
                
            if e % 2 == 0:
                adjust_learning_rate(optimizer, e + 1)
            total_loss = epoch_loss / count
            val_loss = inference(self.model, dev_loader)           
            print("Epoch : ", e+1)
            print("Val loss: ",val_loss.cpu().numpy()[0])
            print("Total loss: ",total_loss.cpu().numpy()[0])
            self.save_model('./offensive-language.pt')



words = np.load("vocab.npy")
tags = np.load("tags.npy")
nwords = words.shape[0]
ntags = tags.shape[0]
UNK = words.tolist().index("<UNK_WORD>")
print ("nwords=%r, ntags=%r" % (nwords, ntags))


model = Model(nwords, ntags, UNK)
model.apply(init_xavier)

print(model)

#model.load_state_dict(torch.load('./xavier_current.pt'))
#print(model)
#xaviermodel.apply(init_xavier)


if torch.cuda.is_available():
    model = model.cuda()
n_epochs = 15
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
trainer = Trainer(model, optimizer)
trainer.run(n_epochs)


#Testing starts here
words = np.load("vocab.npy")
tags = np.load("tags.npy")
nwords = words.shape[0]
ntags = tags.shape[0]
UNK = words.tolist().index("<UNK_WORD>")


model = Model(nwords, 6, UNK)

#model.load_state_dict(torch.load('./stress_predictor.pt'))

if torch.cuda.is_available():
    model = model.cuda()

test_values = dev_file#np.load("~/cmner/NER/test_values.npy")
test_labels = dev_labels#np.load("~/cmner/NER/test_labels.npy")
            
    
    
 
test_dataset = MyDataSet(test_values, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, collate_fn = custom_collate, shuffle = False, num_workers=1, pin_memory=True)
tag = test_function(model, test_loader, tags)   

print(len(test_values))


print(nwords)


#tag_to_ix = np.load("~/cmner/NER/tags_new.npy").item()


print(len(tag_to_ix))

sys.stdout.close()
