# coding: utf-8
from __future__ import print_function
import time
start = time.time()

from collections import Counter, defaultdict
import random
#import sys
import os
import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F


WEMBED_SIZE = 64
CEMBED_SIZE = 50
HIDDEN_SIZE = 128
MLP_SIZE = 64
TIMEOUT = 300000


import sys
sys.stdout = open('loss.log', 'w')

# format of files: each line is "word1|tag2 word2|tag2 ..."
train_folder = "stress_f0"
#train_file = "../data/train.txt"
#dev_file = "../data/dev.txt"

print ("Started ...")

class Vocab:

    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(lambda: len(w2i))
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.items()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(lambda: len(w2i))
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self):
        return len(self.w2i.keys())


def myread():
    sent = []
    all_files = os.listdir(train_folder)
    for file_name in all_files:#[:100]:
        f = open(os.path.join(train_folder, file_name),'r')
        lines = f.readlines()
        cur_sent = []
        for line in lines:
            w = line.strip().split()
            word = w[0]
            tag = w[1]
            tag1 = w[2]
            cur_sent += [(word,tag)]
        f.close()
        #yield cur_sent
        sent += [cur_sent]
    return sent
    #yield sent

def read(fname):
    """
    Read a POS-tagged file where each line is of the form "word1|tag2 word2|tag2 ..."
    Yields lists of the form [(word1,tag1), (word2,tag2), ...]
    """
    with open(fname, "r") as fh:
        for line in fh:
            line = line.strip().split()
            sent = []
            for x in line:
                word, tag = x.rsplit("|", 1)
                if word[0] == "@":
                    word = "<USR>"
                if word.find("http") == 0:
                    word = "<URL>" 
                sent += [(word,tag)]
            yield sent

full_data = list(myread())
train_split_point = int(len(full_data)*4/5)
train = full_data[:train_split_point]
print(len(train))
dev = full_data[train_split_point:]
#dev = list(read(dev_file))
words = []
tags = []
#chars = set()
wc = Counter()
for sent in train:
    for w, p in sent:
        words.append(w)
        tags.append(p)
        wc[w] += 1
        #chars.update(w)
words.append("_UNK_")
#chars.add("_UNK_")
#chars.add("<*>")

vw = Vocab.from_corpus([words])
vt = Vocab.from_corpus([tags])
#vc = Vocab.from_corpus([chars])
UNK = vw.w2i["_UNK_"]
#CUNK = vc.w2i["_UNK_"]
#pad_char = vc.w2i["<*>"]

nwords = vw.size()
ntags = vt.size()
#nchars = vc.size()
print ("nwords=%r, ntags=%r " % (nwords, ntags))
#exit(1)

def get_var(x, volatile=False):
    x = Variable(x, volatile=volatile)
    return x.cuda() if torch.cuda.is_available() else x


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        self.lookup_w = nn.Embedding(nwords, WEMBED_SIZE, padding_idx=UNK)
        #self.lookup_c = nn.Embedding(nchars, CEMBED_SIZE, padding_idx=CUNK)
        self.lstm_f = nn.LSTM(WEMBED_SIZE, HIDDEN_SIZE, dropout=0.5)
        #self.lstm_r = nn.LSTM(WEMBED_SIZE, HIDDEN_SIZE, dropout=0.5)
        #self.lstm_c_f = nn.LSTM(CEMBED_SIZE, WEMBED_SIZE // 2, 1)
        #self.lstm_c_r = nn.LSTM(CEMBED_SIZE, WEMBED_SIZE // 2, 1)
        self.dropout = nn.Dropout(0.5)
        self.proj2 = nn.Linear( HIDDEN_SIZE, ntags)
    def forward(self, words, volatile=False):
        word_ids = []
        rev_word_is = []
        needs_chars = []
        char_ids = []
        for i, w in enumerate(words):
            if wc[w] > 0:
                word_ids.append(vw.w2i[w])
            
            else:
                word_ids.append(UNK)
        
        #rev_word_ids = word_ids[::-1] # reversing word ids
        embeddings_f = self.lookup_w(get_var(torch.LongTensor(word_ids), volatile=volatile))
        #embeddings_r = self.lookup_w(get_var(torch.LongTensor(rev_word_ids), volatile=volatile)) 
      
        embeddings_f = self.lstm_f(embeddings_f.unsqueeze(1))[0]
        #embeddings_f = self.dropout(embeddings_f)
        #embeddings_r = self.lstm_r(embeddings_r.unsqueeze(1))[0]
        #embeddings_r = self.dropout(embeddings_r)
        
        #embeddings = torch.cat([embeddings_f, embeddings_r], dim = 2)
        #embeddings,h = self.lstm(embeddings.unsqueeze(1))
        embeddings = embeddings_f.squeeze(1)
        
        #embeddings = self.proj1(embeddings)
        embeddings = self.proj2(embeddings)
        #print (embeddings.size())
        return embeddings

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] =  param_group['lr']
        #param_group['lr'] =  param_group['lr']  / ( 1 + epoch * np.sqrt(2))


model = Model()
from torch import optim

if torch.cuda.is_available():
    model.cuda()
optimizer = optim.Adam(model.parameters(),lr=0.01)
#scheduler_exp = optim.lr_scheduler.ExponentialLR(optimizer, gamma= np.sqrt(2), last_epoch=-1)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


print("startup time: %r" % (time.time() - start))
start = time.time()
i = all_time = dev_time = all_tagged = this_tagged = this_loss = 0
prev_dev_loss = sys.maxsize
#batch_size = 64

import sys
print("Size of train", len(train))


def evaluate():
   words, golds = zip(*s)
   Y = get_var(torch.LongTensor([vt.w2i[t] for t in golds]))

   model.eval()
   devcount = 0
   dev_loss = 0
   for sent in dev:
                devcount += 1
                words, golds = zip(*sent)
                preds =  model(words)
                #print("I predicted ", preds)
                Y = get_var(torch.LongTensor([vt.w2i[t] for t in golds])) 
                loss1 = F.cross_entropy(preds, Y)
                #print("My loss is ", loss1)
                dev_loss = dev_loss + loss1.data.cpu().numpy()
   dev_loss = dev_loss/devcount
   print ("-------")        
   print("Dev loss: ", dev_loss)
   optimizer.zero_grad()
   model.train()
   return dev_loss
       
for ITER in range(10):
    adjust_learning_rate(optimizer, ITER)
    for param_group in optimizer.param_groups:
         print("Value of LR is: ", param_group['lr'])

    random.shuffle(train)
    batch_count = 0
    optimizer.zero_grad()
    loss = 0
    this_loss = 0
    for itr, s in enumerate(train):
        #if itr % 1000 == 1:
        #print("Processed ", itr, " sequences")
        #print(itr)
        #batch_count += 1
        #i += 1
        words, golds = zip(*s)
        Y = get_var(torch.LongTensor([vt.w2i[t] for t in golds]))

        preds = model(words)
        loss =  F.cross_entropy(preds, Y)
        loss.backward()
        this_loss += loss.data[0]*len(golds)
        this_tagged += len(golds)
        #nn.utils.clip_grad_norm(model.parameters(), 0.25)
        optimizer.step()
        optimizer.zero_grad()
        nn.utils.clip_grad_norm(model.parameters(), 0.25)
        loss = 0
        #print("  Current Loss:", this_loss)
       
    print("epoch %r finished" % ITER)
    print("Train loss: ", this_loss / this_tagged, file=sys.stderr)
    this_loss = this_tagged = 0
    all_time = time.time() - start
    val_loss = evaluate()
    #scheduler.step(val_loss)

sys.stdout.close()
