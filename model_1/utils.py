import jieba
import numpy as np
import os
import json
from torch.autograd import Variable
import torch


class Utils(object):
    def __init__(self, path1, path2):
        self.dic = {}
        self.data_dic = {}
        self.label_dic = {}
        self.process_data(path1,path2)


    def process_data(self, path1, path2):
        data = []
        with open(path1, 'r') as file:
            for line in file:
                _, sen1, sen2 = line.strip().split('\t')
                words1 = [w for w in jieba.cut(sen1) if w.strip()]
                words2 = [w for w in jieba.cut(sen2) if w.strip()]
                words1.append('<CAT>')
                data.append(words1 + words2)
        self.bulid_dic()
        sen2ids = self.sen2id(data)
        labels = []
        with open(path2, 'r') as file:
            for la in file:
                labels.append(la.strip())
        for line in sen2ids:
            self.data_dic[len(self.data_dic)] = line


        for line in labels:
            self.label_dic[len(self.label_dic)] = line


    def bulid_dic(self):
        if os.path.exists('./ckpt/vocab.json'):
            with open('./ckpt/vocab.json', 'r') as file:
                self.dic = json.load(file)
        else:
            with open('./ckpt/word2vec_init/vocab.txt', 'r') as file:
                for line in file:
                    word, count = line.strip().split(' ')
                    if not word==None:
                        self.dic[word] = len(self.dic)
            self.dic['<PAD>'] = len(self.dic)
            self.dic['<UNK>'] = len(self.dic)
            self.dic['<CAT>'] = len(self.dic)
            with open('./ckpt/vocab.json', 'w') as outfile:
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
            while index in sample or self.label_dic[index] == '1':
                index = np.random.randint(0, len(self.label_dic)-1)
            sample.append(index)
            tt = []
            tt.append(self.data_dic[index])
            tt.append(self.label_dic[index])
            data.append(tt)
        for i in range(int(batch_size/2)):
            index = np.random.randint(0, len(self.label_dic)-1)
            while index in sample or self.label_dic[index] == '0':
                index = np.random.randint(0, len(self.label_dic)-1)
            sample.append(index)
            tt = []
            tt.append(self.data_dic[index])
            tt.append(self.label_dic[index])
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



