from gensim.models import Word2Vec
import jieba
import numpy as np

path = 'data/train/balanced_sen'
sentences = []
with open(path, 'r') as file:
    for line in file:
        _, sen1, sen2 = line.strip().split('\t')
        words1 = [w for w in jieba.cut(sen1) if w.strip()]
        sentences.append(words1)
        words2 = [w for w in jieba.cut(sen2) if w.strip()]
        sentences.append(words2)

model = Word2Vec(sentences=sentences, window=3, min_count=1, max_vocab_size=20000, size=300, sg=1, iter=20)
model.wv.save_word2vec_format(fname='model_cnn2/ckpt/word2vec_init/word2vec.txt', fvocab='model_cnn2/ckpt/word2vec_init/vocab.txt')
model.save('model_cnn2/ckpt/word2vec_init/model')
