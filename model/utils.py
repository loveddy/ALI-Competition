import jieba
import numpy as np
import os
import json
import torch
import sys
from tqdm import tqdm

TRAIN_DIC_PATH = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/train/train.json')
TRAIN_LABEL_DIC_PATH = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/train/train_label.json')
VOCAB_DIC = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/vocab.json')
WORD_INIT = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/word2vec_init/vocab.txt')

class Utils(object):
    def __init__(self):
        self.dic = {}
        self.data_dic = {}
        self.label_dic = {}


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
                    temp = []
                    temp.append(words1)
                    temp.append(words2)
                    data.append(temp)
            sen2ids = self.sen2id(data)
            labels = []
            with open(path2, 'r') as file:
                for la in file:
                    labels.append(la.strip())
            for i in tqdm(range(len(sen2ids))):
                self.data_dic[str(len(self.data_dic))] = sen2ids[i]
            for i in tqdm(range(len(labels))):
                self.label_dic[str(len(self.label_dic))] = labels[i]
            with open(TRAIN_DIC_PATH, 'w') as out_1, open(TRAIN_LABEL_DIC_PATH, 'w') as out_2:
                json.dump(self.data_dic, out_1, ensure_ascii=False)
                json.dump(self.label_dic, out_2, ensure_ascii=False)



    def build_dic(self):
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
            with open(VOCAB_DIC, 'w') as outfile:
                json.dump(self.dic, outfile, ensure_ascii=False)

    def sen2id(self, sens):
        sen2ids = []
        for i in tqdm(range(len(sens))):
            line = sens[i]
            temp = []
            temp_1 = []
            temp_2 = []
            for word in line[0]:
                if word in self.dic.keys():
                    temp_1.append(self.dic[word])
                else:
                    temp_1.append(self.dic['<UNK>'])
            for word in line[1]:
                if word in self.dic.keys():
                    temp_2.append(self.dic[word])
                else:
                    temp_2.append(self.dic['<UNK>'])
            temp.append(temp_1)
            temp.append(temp_2)
            sen2ids.append(temp)
        return sen2ids

    def by_score(self, t):
        return len(t[0])

    def by_score_s(self, t):
        return len(t)

    def get_random_batch(self, batch_size, mode='train'):
        sample = []
        data = []
        if mode == 'train':
            for i in range(int(batch_size / 2)):
                index = np.random.randint(0, len(self.label_dic) - 1)
                while index in sample or self.label_dic[str(index)] == '1':
                    index = np.random.randint(0, len(self.label_dic) - 1)
                sample.append(index)
                tt = []
                tt.append(self.data_dic[str(index)][0])
                tt.append(self.data_dic[str(index)][1])
                tt.append(self.label_dic[str(index)])
                data.append(tt)
            for i in range(int(batch_size / 2)):
                index = np.random.randint(0, len(self.label_dic) - 1)
                while index in sample or self.label_dic[str(index)] == '0':
                    index = np.random.randint(0, len(self.label_dic) - 1)
                sample.append(index)
                tt = []
                tt.append(self.data_dic[str(index)][0])
                tt.append(self.data_dic[str(index)][1])
                tt.append(self.label_dic[str(index)])
                data.append(tt)
        else:
            for i in range(int(batch_size)):
                index = np.random.randint(0, len(self.label_dic) - 1)
                while index in sample:
                    index = np.random.randint(0, len(self.label_dic) - 1)
                sample.append(index)
                tt = []
                tt.append(self.data_dic[str(index)][0])
                tt.append(self.data_dic[str(index)][1])
                tt.append(self.label_dic[str(index)])
                data.append(tt)

        data_temp = sorted(data, key=self.by_score, reverse=True)
        batch_sen1 = []
        len_sen1 = []
        sen2 = []
        label = []
        for line in data_temp:
            len_sen1.append(len(line[0]))
            batch_sen1.append(line[0]+[self.dic['<PAD>'] for _ in range(len(data_temp[0][0])-len(line[0]))])
            sen2.append(line[1])
            label.append(line[2])

        sen2_sorted = sorted(sen2, key=self.by_score_s, reverse=True)
        batch_sen2 = []
        len_sen2 = []
        index_list = []


        for line in sen2_sorted:
            len_sen2.append(len(line))
            batch_sen2.append(line+[self.dic['<PAD>'] for _ in range(len(sen2_sorted[0])-len(line))])
            sign = 0
            for i, s in enumerate(sen2):
                if s == line:
                    sign = i
                    break
            index_list.append(sign)


        return torch.tensor(batch_sen1, dtype=torch.long), torch.tensor(batch_sen2, dtype=torch.long), len_sen1, len_sen2,torch.tensor(np.array(label, dtype=int), dtype=torch.long), index_list



