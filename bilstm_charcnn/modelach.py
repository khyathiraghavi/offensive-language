#coding: utf-8
from __future__ import print_function
import time
start = time.time()

from collections import Counter, defaultdict
import random
import sys
import argparse
import numpy
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import csv
import re
'''
parser = argparse.ArgumentParser()
parser.add_argument('CEMBED_SIZE', type=int, help='char embedding size')
parser.add_argument('WEMBED_SIZE', type=int, help='embedding size')
parser.add_argument('HIDDEN_SIZE', type=int, help='hidden size')
parser.add_argument('MLP_SIZE', type=int, help='embedding size')
parser.add_argument('SPARSE', type=int, help='sparse update 0/1')
parser.add_argument('TIMEOUT', type=int, help='timeout in seconds')
parser.add_argument('--CUDA', default=1, type=int)
args = parser.parse_args()
'''
CEMBED_SIZE = 50
HIDDEN_SIZE = 100
MLP_SIZE = 150
TIMEOUT = 300000

# format of files: each line is "word1|tag2 word2|tag2 ..."
train_file = "../data/kaggle/train.csv"
dev_file = "../data/kaggle/dev.csv"
#train_file = "train.txt"
#dev_file = "dev.txt"


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

def myread1(fname):
    with open(train_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if i==0:
                i+=1
                continue
            line=row[1:7]
            sent=line[0].strip()
            sent = re.sub(r'[^\w\s]','',sent).split()
            tag=line[1]
            yield (sent,tag)
def myread(fname):
    with open(fname,'r') as fh:
        for line in fh:
            line = line.strip().split()
            sent = line[:-1]
            tag = line[-1]
            print(sent,tag)
            yield (sent, tag)



def read(fname):
    """
    Read a POS-tagged file where each line is of the form "word1|tag2 word2|tag2 ..."
    Yields lists of the form [(word1,tag1), (word2,tag2), ...]
    """
    with open(fname, "r") as fh:
        for line in fh:
            line = line.strip().split()
            sent = [tuple(x.rsplit("|", 1)) for x in line]
            yield sent

train = list(myread1(train_file))
print(len(train))
#dev = list(train[:int(len(train)/5)])
#train = list(train[int(len(train)/5):])
train = train[0:20]
dev = train[0:20]

words = []
tags = []
chars = set()
wc = Counter()
for sent in train:
    tags.append(sent[1])
    for w in sent[0]:
        words.append(w)
        wc[w] += 1
        chars.update(w)
words.append("_UNK_")
chars.add("_UNK_")
chars.add("<*>")


vw = Vocab.from_corpus([words])
vt = Vocab.from_corpus([tags])
vc = Vocab.from_corpus([chars])
UNK = vw.w2i["_UNK_"]
CUNK = vc.w2i["_UNK_"]
pad_char = vc.w2i["<*>"]

nwords = vw.size()
ntags = vt.size()
nchars = vc.size()
print ("nwords=%r, ntags=%r, nchars=%r" % (nwords, ntags, nchars))


def get_var(x, volatile=False):
    x = Variable(x, volatile=volatile)
    return x.cuda() if torch.cuda.is_available() else x


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.lookup_c = nn.Embedding(nchars, CEMBED_SIZE, padding_idx=CUNK)
        # self.conv13 = nn.Conv1d(CEMBED_SIZE, HIDDEN_SIZE, 3)
        # self.conv14 = nn.Conv1d(CEMBED_SIZE, HIDDEN_SIZE, 4)
        # self.conv15 = nn.Conv1d(CEMBED_SIZE, HIDDEN_SIZE, 5)
        self.conv13 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_SIZE, kernel_size=(CEMBED_SIZE, 3))
        self.conv14 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_SIZE, kernel_size=(CEMBED_SIZE, 4))
        self.conv15 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_SIZE, kernel_size=(CEMBED_SIZE, 5))
        self.lstm = nn.LSTM(3 * HIDDEN_SIZE, HIDDEN_SIZE, 3 ,  batch_first = True, bidirectional = True)
        self.dropout = nn.Dropout(0.4)
        self.proj1 = nn.Linear(2 * HIDDEN_SIZE, MLP_SIZE)
        self.proj2 = nn.Linear(MLP_SIZE, 1)

    def forward(self, words, volatile=False):
        needs_chars = []
        char_ids = []

        for i, w in enumerate(words):
            needs_chars.append(i)
            char_ids.append( [vc.w2i.get(c, CUNK) for c in w] )

        if needs_chars:
            max_len = max(len(x) for x in char_ids)
            if max_len < 5:
                max_len = 5
            fwd_char_ids = [ids + [pad_char for _ in range(max_len - len(ids))] for ids in char_ids]


            embeddings = self.lookup_c(get_var(torch.LongTensor(fwd_char_ids))) #(3,7,50)

            embeddings = torch.transpose(embeddings, 2, 1)
            embeddings = embeddings.unsqueeze(1)
            x2 = self.conv13(embeddings) #(3,100,1,3)
            x2 = F.relu(x2).squeeze(2) #(3,100,3)
            x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)

            x3 = self.conv13(embeddings)
            x3 = F.relu(x3).squeeze(2)
            x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)

            x4 = self.conv13(embeddings)
            x4 = F.relu(x4).squeeze(2)
            x4 = F.max_pool1d(x4, x4.size(2)).squeeze(2)


            embeddings = torch.cat((x2, x3, x4), 1) #(3,300)

        embeddings = embeddings.unsqueeze(0)
        embeddings, h = self.lstm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = torch.transpose(embeddings,2,1)
        embeddings = F.max_pool1d(embeddings, embeddings.size(2)).squeeze(2)

        out = self.proj2(self.proj1(embeddings)).squeeze(0) #(3,4)
        outsig = F.sigmoid(out)
        return outsig


model = Model()
#optimizer = optim.Adam(model.parameters(), lr = 0.00001, weight_decay=0.000001)
#model.load_state_dict(torch.load("./current_model1.pt"))
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr = 0.0001, weight_decay=0.000001)

loss = nn.BCELoss()
#loss = nn.CrossEntropyLoss()

print("startup time: %r" % (time.time() - start))
start = time.time()
i = all_time = dev_time = all_tagged = this_tagged = this_loss = 0

batch_size = 1
log_after = 1
for ITER in range(100):
    count = 0
    epoch_loss = 0
    correct = 0
    total_correct = 0
    #random.shuffle(train)
    batch_count = 0
    
    for s in train:
	batch_count +=1
        i += 1
        #print(i)
	'''
        if i % 500 == 0:
            all_tagged += this_tagged
            this_loss = this_tagged = 0
            all_time = time.time() - start
	'''
        if i % log_after == 0 :  # eval on dev
            fout = open("output"+str(i)+".txt", "w")
            #dev_start = time.time()
            #good_sent = bad_sent = good = bad = 0.0
            for sent in dev:
                #words, golds = zip(*sent)
		words = sent[0]
		tags = sent[1]
		preds = model(words, volatile=True)
            	devgolds = get_var(torch.FloatTensor([1]))
        	if tags == '1' or tags==1:
            		devgolds = get_var(torch.FloatTensor([1]))
        	else:
            		devgolds = get_var(torch.FloatTensor([0]))
		p = preds.data.numpy()[0]
		if float(p) < 0.5:
		    fout.write("0\n")
		else:
		    fout.write("1\n")
	    #dev_loss += loss(preds, devgolds)
            #print ("Dev Loss: ",dev_loss.data.view(-1))
            fout.close()


        # batch / loss
        #words, golds = zip(*s)
        words = s[0]
        tags = s[1]
        count = count + len(words)
        preds = model(words)
        #print (preds)
        #golds = [tags]

        #golds = get_var(torch.LongTensor([0,0]))
        golds = get_var(torch.FloatTensor([0,0]))
        if tags == '1' or tags==1:
            #golds = get_var(torch.LongTensor([1]))
            golds = get_var(torch.FloatTensor([1]))
        else:
            #golds = get_var(torch.LongTensor([0]))
            golds = get_var(torch.FloatTensor([0]))

        Y = golds
        #Y = get_var(torch.LongTensor([vt.w2i[t] for t in golds]))
        #pred = (preds.data.max(1, keepdim=True)[1]).long()
        #predicted = pred.eq(Y.data.view_as(pred))
        #correct += predicted.sum()
        print (preds.size())
        #print (Y.size())
        #print("Y is ",Y)
        #print("preds is",preds)
        #exit(1)
        #loss = F.cross_entropy(preds, Y)
        
        loss1 = loss(preds, Y)
        print ("Train Loss: ",loss1.data.view(-1))
        #exit(1)
        nn.utils.clip_grad_norm(model.parameters(), 0.25)
        # log / optim
        #this_loss += loss.data[0]*len(golds)
        #this_tagged += len(golds)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        epoch_loss = epoch_loss + loss1[0]
    torch.save(model.state_dict(), './current_model1.pt')
    print("epoch %r finished" % ITER)
    print("Epoch loss : ",  epoch_loss /count)
