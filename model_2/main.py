#/usr/bin/env python
#coding=utf-8
import jieba
import sys
import json
import os
from model import GRUEncoder, Classifier
from gensim.models import Word2Vec
import numpy as np
import torch

def build_model(voc_size, emb_dim, hidden_size):
    encoder = GRUEncoder(emb_dim, hidden_size, voc_size, emb_dim)
    classifier = Classifier(hidden_size)
    encoder.load_state_dict(torch.load(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/encoder')))
    classifier.load_state_dict(torch.load(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/classifier')))
    with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/vocab.json'), 'r') as file:
        dic = json.load(file)

    return encoder, classifier, dic

def process(inpath, outpath):
    encoder, classifier, dic = build_model(5569, 128, 256)
    with torch.no_grad():
        with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
            for line in fin:
                lineno, sen1, sen2 = line.strip().split('\t')
                words1 = [w for w in jieba.cut(sen1) if w.strip()]
                words2 = [w for w in jieba.cut(sen2) if w.strip()]
                words1.append('<CAT>')
                temp = words1 + words2
                sen2id = []
                for word in temp:
                    if word in dic.keys():
                        sen2id.append(dic[word])
                    else:
                        sen2id.append(dic['<UNK>'])
                sen = torch.tensor([sen2id], dtype=torch.long)
                output, _ = encoder(sen)
                logit = classifier(torch.mean(output[0], dim=0)).view(1, 2)
                if logit.data[0][0] > logit.data[0][1]:
                    fout.write(lineno + '\t0\n')
                else:
                    fout.write(lineno + '\t1\n')

if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
