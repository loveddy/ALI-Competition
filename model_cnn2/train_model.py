import torch
from utils import Utils
from model import Model
import torch.optim as optim
from gensim.models import Word2Vec
import numpy as np
import os
import sys
import jieba

MAX_LENGTH = 20
def build_model(kernel_size, voc_size, emb_dim):
    model = Model(kernel_size, voc_size, emb_dim)
    if os.path.exists(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/model')):
        model.load_state_dict(torch.load(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/model')))
    else:
        word2vec = Word2Vec.load(
            os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/word2vec_init/model'))
        trained = np.random.random(size=(voc_size, emb_dim))
        trained[:-2][:] = np.array(word2vec.wv. vectors).reshape(voc_size - 2, emb_dim)
        trained[-1][:] = np.zeros(emb_dim).reshape(1, emb_dim)
        model.embedding.weight = torch.nn.Parameter(torch.tensor(trained, dtype=torch.float))
        model.init_parameters()

    return model


def train_iter(model, criterion, optimizer):
    model.train()
    batch_size = 128
    optimizer.zero_grad()
    sens_1, sens_2, label = helper.get_random_batch(batch_size)
    output = model(sens_1, sens_2)
    target = torch.tensor(label, dtype=torch.float)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_test(model):
    with torch.no_grad():
        model.eval()
        batch_size = 128
        sens_1, sens_2, label = helper.get_random_batch(batch_size, mode='test')
        output = model(sens_1, sens_2)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for j in range(batch_size):
            lo = 0
            if output.data[j] < 0.:
                lo = -1
            else:
                lo = 1
            if lo == 1 and label.data[j] == 1:
                TP += 1
            if lo == 1 and label.data[j] == -1:
                FP += 1
            if lo == -1 and label.data[j] == -1:
                TN += 1
            if lo == -1 and label.data[j] == 1:
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


def evaluate(model, helper):
    model.eval()
    with torch.no_grad():
        result = []
        label = []
        with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/test/sens'), 'r') as file_sen, open(
            os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/test/label'), 'r') as file_label:
            for line in file_sen:
                lineno, sen1, sen2 = line.strip().split('\t')
                words1 = [w for w in jieba.cut(sen1) if w.strip()]
                words2 = [w for w in jieba.cut(sen2) if w.strip()]
                sen_id_1 = []
                sen_id_2 = []
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
                if len(sen_id_1) <= MAX_LENGTH:
                    sen_id_1 = sen_id_1 + [helper.dic['<PAD>'] for _ in range(MAX_LENGTH - len(sen_id_1))]
                else:
                    sen_id_1 = sen_id_1[:MAX_LENGTH]
                if len(sen_id_2) <= MAX_LENGTH:
                    sen_id_2 = sen_id_2 + [helper.dic['<PAD>'] for _ in range(MAX_LENGTH - len(sen_id_2))]
                else:
                    sen_id_2 = sen_id_2[:MAX_LENGTH]

                data_1 = torch.tensor(sen_id_1, dtype=torch.long).view(1, -1)
                data_2 = torch.tensor(sen_id_2, dtype=torch.long).view(1, -1)

                output = model(data_1, data_2)
                if output.data[0] < 0.:
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


# def loss_function(logit, label):
#     loss = torch.tensor(data=0, dtype=torch.float)
#     loss.requires_grad_()
#     for i in range(logit.size()[0]):
#         loss_ = loss.clone()
#         if label[i].item() == 0:
#             loss = loss_ + torch.exp(logit[i] * -1.)[0]
#         else:
#             loss = loss_ + torch.exp(logit[i])[0]
#     return loss


helper = Utils()
helper.build_dic()
helper.process_data(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/temp_train/label'),
                    os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/temp_train/sens'))

kernel_size = 2
model = build_model(kernel_size, len(helper.dic), 300)
optimizer = optim.Adam(model.cnn.parameters(), lr=0.0001)

criterion = torch.nn.MSELoss()
loss_list = []
print "epoch    precision    recall    accuracy    F1_score    train_loss"
record = 0.
for i in range(10000):
    loss = train_iter(model, criterion, optimizer)
    loss_list.append(loss)
    if (i + 1) % 1 == 0:
        precision, recall, accuracy, f1_score = train_test(model)
        print str(i + 1) + '        ' + str(precision) + '       ' + str(recall) + '      ' + str(
            accuracy) + '        ' + str(f1_score) + '       ' + str(round(sum(loss_list[-1:]) / 1.0, 4))
    if (i + 1) % 1000 == 0:
        precision, recall, accuracy, f1_score = evaluate(model, helper)
        print(str(i + 1) + '        ' + str(precision) + '      ' + str(recall) + '     ' + str(
            accuracy) + '       ' + str(f1_score) + '            evaluate')
        if f1_score > record:
            torch.save(model.state_dict(),
                       os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/model'))
            record = f1_score
