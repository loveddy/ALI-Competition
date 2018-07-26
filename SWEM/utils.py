import jieba
import numpy as np
import os
import json
import torch
from tqdm import tqdm
import sys

TRAIN_DIC_PATH = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/temp_train/sens.json')
TRAIN_LABEL_DIC_PATH = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/temp_train/label.json')
VOCAB_DIC = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/vocab.json')
WORD_INIT = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/word2vec_init/vocab.txt')
MAX_LENGTH = 20

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
                for line in file:
                    idx, la = line.strip().split('\t')
                    labels.append(la)
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

    # def get_random_batch(self, batch_size, mode='train'):
    #     sample = []
    #     data = []
    #     label = []
    #     if mode == 'train':
    #         for i in range(int(batch_size / 2)):
    #             index = np.random.randint(0, len(self.label_dic) - 1)
    #             while index in sample or self.label_dic[str(index)] == '1':
    #                 index = np.random.randint(0, len(self.label_dic) - 1)
    #             sample.append(index)
    #             tt = self.data_dic[str(index)]
    #             label.append('-1')
    #             data.append(tt)
    #         for i in range(int(batch_size / 2)):
    #             index = np.random.randint(0, len(self.label_dic) - 1)
    #             while index in sample or self.label_dic[str(index)] == '0':
    #                 index = np.random.randint(0, len(self.label_dic) - 1)
    #             sample.append(index)
    #             tt = self.data_dic[str(index)]
    #             label.append('1')
    #             data.append(tt)
    #     if mode == 'test':
    #         for i in range(int(batch_size * 4 / 5)):
    #             index = np.random.randint(0, len(self.label_dic) - 1)
    #             while index in sample or self.label_dic[str(index)] == '1':
    #                 index = np.random.randint(0, len(self.label_dic) - 1)
    #             sample.append(index)
    #             tt = self.data_dic[str(index)]
    #             label.append('-1')
    #             data.append(tt)
    #         for i in range(batch_size - int(batch_size * 4 / 5)):
    #             index = np.random.randint(0, len(self.label_dic) - 1)
    #             while index in sample or self.label_dic[str(index)] == '0':
    #                 index = np.random.randint(0, len(self.label_dic) - 1)
    #             sample.append(index)
    #             tt = self.data_dic[str(index)]
    #             label.append(self.label_dic[str(index)])
    #             data.append(tt)
    #     batch_sen_1 = []
    #     batch_sen_2 = []
    #     len_1 = []
    #     len_2 = []
    #     max_len_1 = max([len(l[0]) for l in data])
    #     max_len_2 = max([len(l[1]) for l in data])
    #     for line in data:
    #         batch_sen_1.append(line[0] + [self.dic['<PAD>'] for _ in range(max_len_1 - len(line[0]))])
    #         len_1.append(len(line[0]))
    #         batch_sen_2.append(line[1] + [self.dic['<PAD>'] for _ in range(max_len_2 - len(line[1]))])
    #         len_2.append(len(line[1]))
    #
    #     return torch.tensor(batch_sen_1, dtype=torch.long), torch.tensor(batch_sen_2, dtype=torch.long), torch.tensor(np.array(label, dtype=int), dtype=torch.long), len_1, len_2
    #

    def get_random_batch(self, batch_size, mode='train'):
        sample = []
        data = []
        label = []
        if mode == 'train':
            for i in range(int(batch_size / 2)):
                index = np.random.randint(0, len(self.label_dic) - 1)
                while index in sample or self.label_dic[str(index)] == '1':
                    index = np.random.randint(0, len(self.label_dic) - 1)
                sample.append(index)
                tt = self.data_dic[str(index)]
                label.append(self.label_dic[str(index)])
                data.append(tt)
            for i in range(int(batch_size / 2)):
                index = np.random.randint(0, len(self.label_dic) - 1)
                while index in sample or self.label_dic[str(index)] == '0':
                    index = np.random.randint(0, len(self.label_dic) - 1)
                sample.append(index)
                tt = self.data_dic[str(index)]
                label.append('1')
                data.append(tt)
        if mode == 'test':
            for i in range(int(batch_size)):
                index = np.random.randint(0, len(self.label_dic) - 1)
                while index in sample:
                    index = np.random.randint(0, len(self.label_dic) - 1)
                sample.append(index)
                tt = self.data_dic[str(index)]
                label.append(self.label_dic[str(index)])
                data.append(tt)
        batch_sen_1 = []
        batch_sen_2 = []
        for line in data:
            if len(line[0]) <= MAX_LENGTH:
                batch_sen_1.append(line[0] + [self.dic['<PAD>'] for _ in range(MAX_LENGTH - len(line[0]))])
            else:
                batch_sen_1.append((line[0][:MAX_LENGTH]))
            if len(line[1]) <= MAX_LENGTH:
                batch_sen_2.append(line[1] + [self.dic['<PAD>'] for _ in range(MAX_LENGTH - len(line[1]))])
            else:
                batch_sen_2.append((line[1][:MAX_LENGTH]))

        return torch.tensor(batch_sen_1, dtype=torch.long), torch.tensor(batch_sen_2, dtype=torch.long), torch.tensor(np.array(label, dtype=int), dtype=torch.long)


