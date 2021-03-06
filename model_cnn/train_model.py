import torch
from utils import Utils
from model import CNNEncoder, Classifier
import torch.optim as optim
from gensim.models import Word2Vec
import numpy as np
import os
import sys

def build_model(out_size, kernel_size, voc_size, emb_dim):
    encoder = CNNEncoder(out_size, kernel_size, voc_size, emb_dim)
    classifier = Classifier(out_size)
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

def train_iter(helper, encoder, classifier, encoder_optimizer, classifier_optimizer, kernel_size):
    encoder.train()
    classifier.train()
    batch_size = 128
    encoder_optimizer.zero_grad()
    classifier_optimizer.zero_grad()
    sens_1, sens_2, length_1, length_2, label = helper.get_random_batch(batch_size)
    output_1 = encoder(sens_1)
    output_2 = encoder(sens_2)
    loss = 0.
    for j in range(batch_size):
        end_1 = length_1[j] - kernel_size + 1
        end_2 = length_2[j] - kernel_size + 1
        if end_1 <= 0:
            end_1 = 1
        if end_2 <= 0:
            end_2 = 1
        logit = classifier(torch.mean(output_1[j, :, :end_1], dim=1).view(1, -1), torch.mean(output_2[j, :, :end_2], dim=1).view(1,-1))
        loss += criterion(logit, label[j].view(1))
    loss.backward()
    encoder_optimizer.step()
    classifier_optimizer.step()
    return loss.item() / batch_size

def evaluate(encoder, classifier, helper):
    with torch.no_grad():
        encoder.eval()
        classifier.eval()
        batch_size = 128
        sens_1, sens_2, length_1, length_2, label = helper.get_random_batch(batch_size)
        output_1 = encoder(sens_1)
        output_2 = encoder(sens_2)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for j in range(batch_size):
            end_1 = length_1[j] - kernel_size + 1
            end_2 = length_2[j] - kernel_size + 1
            if end_1 <= 0:
                end_1 = 1
            if end_2 <= 0:
                end_2 = 1
            logit = classifier(torch.mean(output_1[j, :, :end_1], dim=1).view(1, -1),
                                   torch.mean(output_2[j, :, :end_2], dim=1).view(1, -1))
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
helper.process_data(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/train_add'), os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/label_add'))

kernel_size = 3
encoder, classifier = build_model(128, kernel_size, len(helper.dic), 128)



encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)

criterion = torch.nn.CrossEntropyLoss()
loss_list = []

print "epoch    precision    recall    accuracy    F1_score    train_loss"
for i in range(20000):
    loss = train_iter(helper, encoder, classifier, encoder_optimizer, classifier_optimizer, kernel_size)
    loss_list.append(loss)
    if (i + 1) % 100 == 0:
        precision, recall, accuracy, f1_score = evaluate(encoder, classifier, helper)
        print str(i+1) + '        ' + str(precision) + '       ' + str(recall) + '      ' + str(accuracy) + '        ' + str(f1_score) + '       ' + str(round(sum(loss_list[-100:]) / 100.0, 4))
        torch.save(encoder.state_dict(), os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/encoder'))
        torch.save(classifier.state_dict(), os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'ckpt/classifier'))

