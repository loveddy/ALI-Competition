#/usr/bin/env python
#coding=utf-8
import jieba
import sys
import json
import os
from model import Model
import torch


MAX_LENGTH = 20
def build_model(emb_dim):
    with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/vocab.json'), 'r') as file:
        dic = json.load(file)
    voc_size = len(dic)
    model = Model(MAX_LENGTH, voc_size, emb_dim)
    model.load_state_dict(torch.load(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/model')))
    return model, dic

def process(inpath, outpath):
    model, dic = build_model(300)
    with torch.no_grad():
        with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
            for line in fin:
                lineno, sen1, sen2 = line.strip().split('\t')
                words1 = [w for w in jieba.cut(sen1) if w.strip()]
                words2 = [w for w in jieba.cut(sen2) if w.strip()]
                sen_id_1 = []
                sen_id_2 = []
                for word in words1:
                    if word in dic.keys():
                        sen_id_1.append(dic[word])
                    else:
                        sen_id_1.append(dic['<UNK>'])
                for word in words2:
                    if word in dic.keys():
                        sen_id_2.append(dic[word])
                    else:
                        sen_id_2.append(dic['<UNK>'])
                if len(sen_id_1) <= MAX_LENGTH:
                    sen_id_1 = sen_id_1 + [dic['<PAD>'] for _ in range(MAX_LENGTH - len(sen_id_1))]
                else:
                    sen_id_1 = sen_id_1[:MAX_LENGTH]
                if len(sen_id_2) <= MAX_LENGTH:
                    sen_id_2 = sen_id_2 + [dic['<PAD>'] for _ in range(MAX_LENGTH - len(sen_id_2))]
                else:
                    sen_id_2 = sen_id_2[:MAX_LENGTH]

                data_1 = torch.tensor(sen_id_1, dtype=torch.long).view(1, -1)
                data_2 = torch.tensor(sen_id_2, dtype=torch.long).view(1, -1)

                output = model(data_1, data_2)
                if output.data[0][0] > output.data[0][1]:
                    fout.write(lineno + '\t0\n')
                else:
                    fout.write(lineno + '\t1\n')

if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
