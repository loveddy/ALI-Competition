#/usr/bin/env python
#coding=utf-8
import jieba
import sys
import json
import os
from model import GRUEncoder, Classifier
import torch

def build_model(emb_dim, hidden_size):
    with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/vocab.json'), 'r') as file:
        dic = json.load(file)
    voc_size = len(dic)
    encoder = GRUEncoder(emb_dim, hidden_size, voc_size, emb_dim)
    classifier = Classifier(hidden_size)
    encoder.load_state_dict(torch.load(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/encoder')))
    return encoder, classifier, dic

def process(inpath, outpath):
    encoder, classifier, dic = build_model(128, 128)
    with torch.no_grad():
        with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
            for line in fin:
                lineno, sen1, sen2 = line.strip().split('\t')
                words1 = [w for w in jieba.cut(sen1) if w.strip()]
                words2 = [w for w in jieba.cut(sen2) if w.strip()]
                sen_id_1 = []
                sen_id_2 =[]
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
                data_1 = torch.tensor(sen_id_1, dtype=torch.long).view(1, -1)
                data_2 = torch.tensor(sen_id_2, dtype=torch.long).view(1, -1)
                _, _, output_1 = encoder(data_1)
                _, _, output_2 = encoder(data_2)
                logit = classifier(output_1, output_2)
                if logit.data[0][0] > logit.data[0][1]:
                    fout.write(lineno + '\t0\n')
                else:
                    fout.write(lineno + '\t1\n')

if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
