from gensim.models import Word2Vec
import jieba


path = 'data/bag_train/sens_1'
sentences = []
with open(path, 'r') as file:
    for line in file:
        _, sen1, sen2 = line.strip().split('\t')
        words1 = [w for w in jieba.cut(sen1) if w.strip()]
        sentences.append(words1)
        words2 = [w for w in jieba.cut(sen2) if w.strip()]
        sentences.append(words2)
        print words1, words2

model = Word2Vec(sentences=sentences, window=4, min_count=1, max_vocab_size=20000, size=50, sg=0, iter=100)
model.wv.save_word2vec_format(fname='bagmodel_1/model_cnn_1/ckpt/word2vec_init/word2vec.txt', fvocab='bagmodel_1/model_cnn_1/ckpt/word2vec_init/vocab.txt')
model.save('bagmodel_1/model_cnn_1/ckpt/word2vec_init/model')
