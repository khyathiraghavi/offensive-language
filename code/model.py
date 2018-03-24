# coding: utf-8
import time
start = time.time()

from collections import Counter, defaultdict
import random
import sys
import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import csv

train_file = "../data/kaggle/train.csv"
dev_file = "../data/kaggle/dev.csv"

#hyperparameters
WEMBED_SIZE = 200
CEMBED_SIZE = 200

'''
"id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
"0000997932d777bf","Explanation
Why the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove the template from the talk page since I'm retired now.89.205.38.27",0,0,0,0,0,0
'''

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



def read(fname):
    lines = []
    with open(train_file, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
	for row in reader:
	    if i==0:
		i+=1
		continue
	    #print row[1:7]
            lines.append(row[1:7])
	    #raw_input()
    return lines


train_lists = read(train_file)
dev_lists = read(dev_file)
words = []
tags = []
chars = set()
wc = Counter()
for each_list in train_lists:
    cur_words = each_list[0].strip().split()
    words += cur_words
    tags.append(each_list[1])
    for word in cur_words:
        wc[word] += 1
        chars.update(word)

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
nchars = vc.size()
print "nwords=%r, ntags=%r, nchars=%r" % (vw.size(), vt.size(), vc.size())


def get_var(x, volatile=False):
    x = Variable(x, volatile=volatile)
    return x.cuda() if torch.cuda.is_available() else x


CONV1_SIZE = 100
CONV2_SIZE = 100
CONV3_SIZE = 100
HIDDEN_SIZE = 128

class Model(nn.Module):
    def __init__(self):
	super(Model, self).__init__()
        #embedding
	self.lookup_w = nn.Embedding(nwords, WEMBED_SIZE)
	self.lookup_c = nn.Embedding(nchars, CEMBED_SIZE)
	#convolution
	self.conv13 = nn.Conv1d(CEMBED_SIZE, 100, kernel_size=3)
	self.conv14 = nn.Conv1d(100, 100, kernel_size=4)
	self.conv15 = nn.Conv1d(100, 100, kernel_size=5)
	#lstm
	self.lstm = nn.LSTM(300  ,HIDDEN_SIZE,bidirectional=True)
	#fc
	self.proj1 = nn.Linear(HIDDEN_SIZE, 2)

    def forward(self, words):
	word_ids = []
        needs_chars = []
        char_ids = []

	for i, w in enumerate(words):
            #word_ids.append(vw.w2i[w])
            needs_chars.append(i)
            char_ids.append([pad_char] + [vc.w2i.get(c, CUNK) for c in w] + [pad_char])

	if needs_chars:
	    max_len = max(len(x) for x in char_ids)
            if max_len < 5:
                max_len = 5
            fwd_char_ids = [ids + [pad_char for _ in range(max_len - len(ids))] for ids in char_ids]

	embeddings = self.lookup_c(get_var(torch.LongTensor(fwd_char_ids)))
	x2 = self.conv13(embeddings)
	x2 = F.relu(x2)
	x2 = F.max_pool1d(x2, 3)

	x3 = self.conv14(embeddings)
	x3 = F.relu(x3)
	x3 = F.max_pool1d(x3, 4)
	
	x4 = self.conv15(embeddings)
	x4 = F.relu(x4)
	x4 = F.max_pool1d(x4, 5)

	embeddings = torch.cat((x2,x3,x4))

	h = self.lstm(embeddings)
	return self.proj1(h)


model = Model()
if torch.cuda.is_available():
	model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(200):
	epoch_loss = 0
	random.shuffle(train_lists)
	for el in train_lists:
		words = el[0].strip().split()
		pred = model(words)
		golds = [el[1]]
		Y = get_var(torch.LongTensor([vt.w2i[t] for t in golds]))
		epoch_loss = F.cross_entropy(pred, Y)
		epoch_loss.backward()
		optimizer.zero_grad()
		optimizer.step()





















	






