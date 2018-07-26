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
    classifier = Classifier()
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

def train_iter(helper, encoder, classifier, encoder_optimizer):
    encoder.train()
    classifier.train()
    batch_size = 128
    encoder_optimizer.zero_grad()

    sens_1, sens_2, length_1, length_2, label = helper.get_random_batch(batch_size)
    output_1 = encoder(sens_1)
    output_2 = encoder(sens_2)
    logit = classifier(output_1, output_2)
    target = torch.tensor(label.view(-1, 1), dtype=torch.float)
    loss = criterion(logit, target)
    loss.backward()
    encoder_optimizer.step()

    return loss.item()

def evaluate(encoder, classifier, criterion, helper):
    with torch.no_grad():
        encoder.eval()
        classifier.eval()
        batch_size = 64
        sens_1, sens_2, length_1, length_2, label = helper.get_random_batch(batch_size)
        output_1 = encoder(sens_1)
        output_2 = encoder(sens_2)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        logit = classifier(output_1, output_2)
        target = torch.tensor(label.view(-1, 1), dtype=torch.float)
        loss = criterion(logit, target)
        for j in range(batch_size):
            lo = 0
            if logit.data[j][0] > 0.5:
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
        return round(precision, 3), round(recall, 3), round(accuracy, 3), round(f1_score, 3), round(loss.item(), 3)

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
helper.process_data(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/test'), os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/test_label'))

kernel_size = 3
encoder, classifier = build_model(128, kernel_size, len(helper.dic), 128)



encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0001)

criterion = torch.nn.MSELoss()
loss_list = []

print "epoch    precision    recall    accuracy    F1_score    eval_loss    train_loss"
for i in range(400000):
    loss = train_iter(helper, encoder, classifier, encoder_optimizer)
    loss_list.append(loss)
    if (i + 1) % 10 == 0:
        precision, recall, accuracy, f1_score, loss_ = evaluate(encoder, classifier, criterion, helper)

        print str(i+1) + '        ' + str(precision) + '       ' + str(recall) + '      ' + str(accuracy) + '        ' + str(f1_score) + '        ' + str(loss_) + '       ' + str(round(sum(loss_list[-10:]) / 10.0, 4))
        torch.save(encoder.state_dict(), os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/encoder'))
        torch.save(classifier.state_dict(), os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'ckpt/classifier'))

