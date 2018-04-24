import numpy as np
import re
import string
import torchtext.vocab as vocab
import torch
from collections import *
import os
import csv
import sys
remove = string.punctuation
pattern = r"[{}]".format(remove) # create the pattern
emoji_pattern = r'/[U0001F601-U0001F64F]/u'


def read(filename):
    f = open(filename,"r")
    first = True
    sentences = []
    labels = []
    sentence = []
    LINE = []
    label = []
    i = 1
    embeddings = []
    embedding = []
    for line in f:
        words = [x.lower() for x in line.strip().split("\t")]
        if first == False and words[2] == "0":
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
        sentence += [words[4]]
        if len(words[5].strip()) == 1:
            label += [words[5]]
        else:
            label += [words[5]]
        first = False
    embeddings.append(embedding)
    sentences.append(sentence)
    labels.append(label)
    f.close()
    return sentences, labels



def myread(train_folder):
    sentences = []
    labels = []
    all_files = os.listdir(train_folder)
    for file_name in all_files:
        f = open(os.path.join(train_folder, file_name),'r')
        lines = f.readlines()
        cur_sent = []
        cur_tags = []
        for line in lines:
            w = line.strip().split()
            word = w[0]
            tag = w[1]
            tag1 = w[2]
            cur_sent += [word]
            cur_tags += [tag]
        f.close()
        sentences.append(cur_sent)
        labels.append(map(int,cur_tags))
    return sentences, labels

def myread1(fname):
    sentences=[]
    labels=[]
    both=[]
    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if i==0:
                i+=1
                continue
            line=row[1:]
            sent=line[0].strip()
            sent = re.sub(r'[^\w\s]','',sent).split()
            tag=line[1:]
            sentences.append(sent)
            labels.append(tag)
            both.append([sent,tag])
    idx = np.random.permutation(len(both))
    x,y = np.array(sentences)[idx], np.array(labels)[idx]
    z = np.array(both)[idx]
    z=z.tolist()
    for i in z:
        print(i)
    return x.tolist(), y.tolist()

#dev_sentences, dev_labels = myread("stress_f0")
#train_sentences, train_labels = myread("stress_f0")
train_sentences, train_labels = myread1("../data/kaggle/train.csv")
vocab_set = []
char_set = set()
tag_set = []
for i in range(len(train_sentences)):
    line = " ".join(train_sentences[i])
    vocab_set += train_sentences[i]
    for w in train_sentences[i]:
        for c in w: 
            char_set.add(c)
    labels = " ".join(train_labels[i])
    tag_set += labels.strip().split()
    
vocab_set = set(vocab_set)
tag_set = set(tag_set)
unknown_word = torch.randn([100])
USR = torch.randn([100])
URL = torch.randn([100])
HASHTAG = torch.randn([100])
PUNCT = torch.randn([100])

       
vocab_set.add("<UNK_WORD>")
char_set.add("<UNK_CHAR>")
#char_set.add("<*>")

vocab_set = sorted(vocab_set)
tag_set = sorted(tag_set)
char_set = sorted(char_set)
np.save("vocab.npy",vocab_set)
np.save("tags.npy",tag_set)
c2i = {char_set[i]:i for i in range(len(char_set))}
i2c = {i:char_set[i] for i in range(len(char_set))}
v2i = {vocab_set[i]:i for i in range(len(vocab_set))}
i2v = {i:vocab_set[i] for i in range(len(vocab_set))}
t2i = {tag_set[i]:(i) for i in range(len(tag_set))}
i2t = {(i):tag_set[i] for i in range(len(tag_set))}


train_words = []
train_chars = []
train_label = []
UNK_WORD = v2i['<UNK_WORD>']
UNK_CHAR = c2i['<UNK_CHAR>']
#pad_char = c2i['<*>']

for i in range(len(train_sentences)):
    line = train_sentences[i]
    label = " ".join(train_labels[i])
    lens = [len(w) for w in line]
    max_len = max(lens)
    train_words.append(np.array([v2i.get(w,UNK_WORD)  for w in train_sentences[i]]))
    train_label.append(np.array([t2i[w] for w in label.strip().split()]))
    x=[]
    for w in train_sentences[i]:
        for c in w:
            x.append([c2i.get(c,UNK_CHAR)])
    train_chars.append(np.array(x))
    #train_chars.append(np.array([c2i.get(c,UNK_CHAR)  for c in for w in train_sentences[i]]))
np.save("train_words.npy",train_words)
np.save("train_chars.npy",train_chars)
np.save("train_labels.npy",train_label)
np.save("tags_new.npy", t2i)
