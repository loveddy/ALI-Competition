import jieba
import numpy as np
import os
import json
from torch.autograd import Variable
import torch
from tqdm import tqdm
import sys

TRAIN_DIC_PATH = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/train.json')
TRAIN_LABEL_DIC_PATH = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/train_label.json')
VOCAB_DIC = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/vocab.json')
WORD_INIT = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/word2vec_init/vocab.txt')

class Utils(object):
    def __init__(self, path1, path2):
        self.dic = {}
        self.data_dic = {}
        self.label_dic = {}
        self.process_data(path1, path2)
        self.bulid_dic()


    def process_data(self, path1, path2):
        if os.path.exists(TRAIN_DIC_PATH) and os.path.exists(TRAIN_LABEL_DIC_PATH):
            with open(TRAIN_DIC_PATH, 'r') as data_flie, open(TRAIN_LABEL_DIC_PATH, 'r') as label_file:
                self.data_dic = json.load(data_flie)
                self.label_dic = json.load(label_file)
        else:
            data = []
            with open(path1, 'r') as file:
                for line in file:
                    _, sen1, sen2 = line.strip().split('\t')
                    words1 = [w for w in jieba.cut(sen1) if w.strip()]
                    words2 = [w for w in jieba.cut(sen2) if w.strip()]
                    words1.append('<CAT>')
                    data.append(words1 + words2)
            sen2ids = self.sen2id(data)
            labels = []
            with open(path2, 'r') as file:
                for la in file:
                    labels.append(la.strip())
            for i in tqdm(range(len(sen2ids))):
                self.data_dic[len(self.data_dic)] = sen2ids[i]
            for i in tqdm(range(len(labels))):
                self.label_dic[len(self.label_dic)] = labels[i]
            with open(TRAIN_DIC_PATH, 'w') as out_1, open(TRAIN_LABEL_DIC_PATH, 'w') as out_2:
                json.dump(self.data_dic, out_1, ensure_ascii=False)
                json.dump(self.label_dic, out_2, ensure_ascii=False)



    def bulid_dic(self):
        if os.path.exists(VOCAB_DIC):
            with open(VOCAB_DIC, 'r') as file:
                self.dic = json.load(file)
        else:
            with open(WORD_INIT, 'r') as file:
                for line in file:
                    word, count = line.strip().split(' ')
                    if not word==None:
                        self.dic[word] = len(self.dic)
            self.dic['<PAD>'] = len(self.dic)
            self.dic['<UNK>'] = len(self.dic)
            self.dic['<CAT>'] = len(self.dic)
            with open(VOCAB_DIC, 'w') as outfile:
                json.dump(self.dic, outfile, ensure_ascii=False)

    def sen2id(self, sens):
        sen2ids = []
        for line in sens:
            temp = []
            for word in line:
                if word in self.dic.keys():
                    temp.append(self.dic[word])
                else:
                    temp.append(self.dic['<UNK>'])
            sen2ids.append(temp)
        return sen2ids

    def by_score(self, t):
        return len(t[0])

    def get_random_batch(self, batch_size):
        sample = []
        data = []
        for i in range(int(batch_size/2)):
            index = np.random.randint(0, len(self.label_dic)-1)
            while index in sample or self.label_dic[str(index)] == '1':
                index = np.random.randint(0, len(self.label_dic)-1)
            sample.append(index)
            tt = []
            tt.append(self.data_dic[str(index)])
            tt.append(self.label_dic[str(index)])
            data.append(tt)
        for i in range(int(batch_size/2)):
            index = np.random.randint(0, len(self.label_dic)-1)
            while index in sample or self.label_dic[str(index)] == '0':
                index = np.random.randint(0, len(self.label_dic)-1)
            sample.append(index)
            tt = []
            tt.append(self.data_dic[str(index)])
            tt.append(self.label_dic[str(index)])
            data.append(tt)
        data_temp = sorted(data, key=self.by_score, reverse=True)
        batch_sen = []
        len_sen = []
        label = []
        for line in data_temp:
            len_sen.append(len(line[0]))
            batch_sen.append(line[0]+[self.dic['<PAD>'] for _ in range(len(data_temp[0][0])-len(line[0]))])
            label.append(line[1])
        return torch.tensor(batch_sen, dtype=torch.long), len_sen, torch.tensor(np.array(label, dtype=int), dtype=torch.long)



