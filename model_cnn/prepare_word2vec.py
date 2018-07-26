from gensim.models import Word2Vec
import jieba
import numpy as np

path = '../data/train_add'
sentences = []
with open(path, 'r') as file:
    for line in file:
        _, sen1, sen2 = line.strip().split('\t')
        words1 = [w for w in jieba.cut(sen1) if w.strip()]
        sentences.append(words1)
        words2 = [w for w in jieba.cut(sen2) if w.strip()]
        sentences.append(words2)

model = Word2Vec(sentences=sentences, window=5, min_count=1, max_vocab_size=10000, size=128, sg=0, iter=10)
model.wv.save_word2vec_format(fname='./ckpt/word2vec_init/word2vec.txt', fvocab='./ckpt/word2vec_init/vocab.txt')
model.save('./ckpt/word2vec_init/model')
# print(model.wv.similar_by_word('周岁'))
# model = Word2Vec.load('./ckpt/model')
# print(np.array(model.wv.vectors))
# print(model.vocabulary.sorted_vocab)