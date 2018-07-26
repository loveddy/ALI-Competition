import torch
from utils import Utils
from model import GRUEncoder, Classifier, ContrastiveLoss
import torch.optim as optim
import os
import sys
from gensim.models import Word2Vec
import numpy as np
import jieba

def build_model(voc_size, emb_dim, hidden_size):
    encoder = GRUEncoder(emb_dim, hidden_size, voc_size, emb_dim)
    classifier = Classifier(hidden_size)
    if os.path.exists(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/encoder')):
        encoder.load_state_dict(torch.load(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'ckpt/encoder')))
    else:
        word2vec = Word2Vec.load(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'ckpt/word2vec_init/model'))

        trained = np.random.random(size=(voc_size, emb_dim))
        trained[:-2][:] = np.array(word2vec.wv.vectors).reshape(voc_size-2, emb_dim)
        encoder.embedding.weight = torch.nn.Parameter(torch.tensor(trained, dtype=torch.float))
    if os.path.exists(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/classifier')):
        classifier.load_state_dict(torch.load(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'ckpt/classifier')))

    return encoder, classifier

def train_iter(helper, encoder, classifier, encoder_optimizer, classifier_optimizer):
    encoder.train()
    classifier.train()
    batch_size = 128
    encoder_optimizer.zero_grad()
    classifier_optimizer.zero_grad()
    sens1, sens2, length1, length2, label, index = helper.get_random_batch(batch_size)
    target = torch.tensor(label.view(-1, 1), dtype=torch.float)
    _, hidden1, cell1 = encoder(sens1, length1)
    _, hidden2, cell2 = encoder(sens2, length2)
    loss = 0
    for j in range(batch_size):
        logit = classifier(cell1[0, index[j]], cell2[0, j])
        loss += criterion(logit, label[j].view(1))
    loss.backward()
    encoder_optimizer.step()
    classifier_optimizer.step()
    return loss.item() / batch_size

def evaluate(encoder, classifier, helper):
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        result = []
        label = []
        with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/test'), 'r') as file_sen, open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/test_label'), 'r') as file_label:
            for line in file_sen:
                lineno, sen1, sen2 = line.strip().split('\t')
                words1 = [w for w in jieba.cut(sen1) if w.strip()]
                words2 = [w for w in jieba.cut(sen2) if w.strip()]
                sen_id_1 = []
                sen_id_2 =[]
                for word in words1:
                    if word in helper.dic.keys():
                        sen_id_1.append(helper.dic[word])
                    else:
                        sen_id_1.append(helper.dic['<UNK>'])
                for word in words2:
                    if word in helper.dic.keys():
                        sen_id_2.append(helper.dic[word])
                    else:
                        sen_id_2.append(helper.dic['<UNK>'])
                data_1 = torch.tensor(sen_id_1, dtype=torch.long).view(1, -1)
                data_2 = torch.tensor(sen_id_2, dtype=torch.long).view(1, -1)
                _, _, output_1 = encoder(data_1)
                _, _, output_2 = encoder(data_2)
                logit = classifier(output_1, output_2)
                if logit.data[0][0] > logit.data[0][1]:
                    result.append("0")
                else:
                    result.append("1")
            for la in file_label:
                label.append(la.strip().split('\t')[1])
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for j in range(len(label)):
        if result[j] == "1" and label[j] == "1":
            TP += 1
        if result[j] == "1" and label[j] == "0":
            FP += 1
        if result[j] == "0" and label[j] == "0":
            TN += 1
        if result[j] == "0" and label[j] == "1":
            FN += 1
    precision = 0.0
    if not (TP + FP) == 0:
        precision = TP / ((TP + FP) * 1.0)
    recall = 0.0
    if not (TP + FN) == 0:
        recall = TP / ((TP + FN) * 1.0)
    accuracy = (TP + TN) / ((TP + FP + TN + FN) * 1.0)
    f1_score = 0.0
    if not (precision + recall) == 0:
        f1_score = 2 * precision * recall / (precision + recall)
    return round(precision, 3), round(recall, 3), round(accuracy, 3), round(f1_score, 3)

def test(encoder, classifier, helper):
    with torch.no_grad():
        encoder.eval()
        classifier.eval()
        batch_size = 128
        sens1, sens2, length1, length2, label, index = helper.get_random_batch(batch_size)
        _, hidden1, cell1 = encoder(sens1, length1)
        _, hidden2, cell2 = encoder(sens2, length2)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for j in range(batch_size):
            logit = classifier(cell1[0, index[j]], cell2[0, j])
            lo = 0
            if logit.data[0][0] > logit.data[0][1]:
                lo = 0
            else:
                lo = 1
            if lo == 1 and label.data[j] == 1:
                TP += 1
            if lo == 1 and label.data[j] == 0:
                FP += 1
            if lo == 0 and label.data[j] == 0:
                TN += 1
            if lo == 0 and label.data[j] == 1:
                FN += 1
        precision = 0.0
        if not (TP + FP) == 0:
            precision = TP / ((TP + FP) * 1.0)
        recall = 0.0
        if not (TP + FN) == 0:
            recall = TP / ((TP + FN) * 1.0)
        accuracy = (TP + TN) / ((TP + FP + TN + FN) * 1.0)
        f1_score = 0.0
        if not (precision + recall) == 0:
            f1_score = 2 * precision * recall / (precision + recall)
        return round(precision, 3), round(recall, 3), round(accuracy, 3), round(f1_score, 3)


helper = Utils()
helper.build_dic()
helper.process_data(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/train/train_add'), os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/train/label_add'))

encoder, classifier = build_model(len(helper.dic), 128, 128)



encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0001)
classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.0001)

criterion = torch.nn.CrossEntropyLoss()
loss_list = []
print "epoch    precision    recall    accuracy    F1_score    train_loss"
eval_f1 = 0.
not_break = True
for i in range(20000):
    if not_break:
        loss = train_iter(helper, encoder, classifier, encoder_optimizer, classifier_optimizer)
        loss_list.append(loss)

        if (i + 1) % 100 == 0:
            precision, recall, accuracy, f1_score = test(encoder, classifier, helper)
            print(str(i + 1) + '        ' + str(precision) + '      ' + str(recall) + '     ' + str(
                accuracy) + '       ' + str(f1_score) + '       ' + str(round(sum(loss_list[-100:]) / 100.0, 4)))
            # torch.save(encoder.state_dict(), os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/encoder'))
            # torch.save(classifier.state_dict(),
            #            os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/classifier'))
        if (i + 1) % 200 == 0:
            precision, recall, accuracy, f1_score = evaluate(encoder, classifier, helper)

            if f1_score > eval_f1:
                print(str(i + 1) + '        ' + str(precision) + '      ' + str(recall) + '     ' + str(
                    accuracy) + '       ' + str(f1_score))
                torch.save(encoder.state_dict(),
                           os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/encoder'))
                torch.save(classifier.state_dict(),
                           os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/classifier'))
                eval_f1 = f1_score
            elif f1_score < eval_f1:
                not_break = False
    else:
        break

