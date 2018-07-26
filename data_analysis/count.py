import random
# import jieba
# with open('data/train/balanced_label', 'r') as file:
#     one = 0
#     zero = 0
#     for line in file:
#         idx, label = line.strip().split('\t')
#         if label == '1':
#             one +=1
#         if label == '0':
#             zero +=1
# print(one)
# print(zero)
# with open('data/test/test_label', 'r') as file, open('data/test/label', 'w') as out:
#     count = 1
#     for line in file:
#         if count > 2046:
#             out.write(str(count) + '\t' + line.strip() + '\n')
#             count += 1
#         else:
#             out.write(line)
#             count += 1
# with open('data/train/label_add') as file:
#     count_1 = 0
#     count_0 = 0
#     for line in file:
#         l = line.strip()
#         if l == '0':
#             count_0 +=1
#         if l == '1':
#             count_1 +=1
#     print count_0,count_1,count_0/(count_1*1.0)
# sen_1 = []
# sen_2 = []
# label = []
# with open('data/raw/atec_nlp_sim_train', 'r') as data_1, open('data/raw/atec_nlp_sim_train_add', 'r') as data_2:
#     for line in data_1:
#         id, sen1, sen2, l = line.strip().split('\t')
#         sen_1.append(sen1)
#         sen_2.append(sen2)
#         label.append(l)
#     for line in data_2:
#         id, sen1, sen2, l = line.strip().split('\t')
#         sen_1.append(sen1)
#         sen_2.append(sen2)
#         label.append(l)
# train_num = int(len(label) * 0.8)
# test_num = int(len(label) * 0.1)
# val_num = len(label) - train_num - test_num
#
# sen_1_p = []
# sen_1_n = []
# sen_2_p = []
# sen_2_n = []
# label_p = []
# label_n = []
# p = 0
# n = 0
# for i in range(len(label)):
#     if label[i] == '0':
#         sen_1_n.append(sen_1[i])
#         sen_2_n.append(sen_2[i])
#         label_n.append(label[i])
#         n += 1
#     else:
#         sen_1_p.append(sen_1[i])
#         sen_2_p.append(sen_2[i])
#         label_p.append(label[i])
#         p += 1
# ratio_n = n / (p * 1. + n)
# ratio_p = p / (p * 1. + n)
# train_num_p = int(p * 0.8)
# train_num_n = int(n * 0.8)
# val_num_p = int(p * 0.1)
# val_num_n = int(n * 0.1)
# test_num_p = p - train_num_p - val_num_p
# test_num_n = n - train_num_n - val_num_n
# sen1_train = []
# sen2_train = []
# sen1_test = []
# sen2_test = []
# sen1_val = []
# sen2_val = []
# label_train = []
# label_test = []
# label_val = []
#
# for i in range(p):
#     if i <val_num_p and i >= 0:
#         sen1_val.append(sen_1_p[i])
#         sen2_val.append(sen_2_p[i])
#         label_val.append(label_p[i])
#     elif i >= val_num_p and i < test_num_p + val_num_p:
#         sen1_test.append(sen_1_p[i])
#         sen2_test.append(sen_2_p[i])
#         label_test.append(label_p[i])
#     elif i >=val_num_p + test_num_p:
#         sen1_train.append(sen_1_p[i])
#         sen2_train.append(sen_2_p[i])
#         label_train.append(label_p[i])
# for i in range(n):
#     if i <val_num_n and i >= 0:
#         sen1_val.append(sen_1_n[i])
#         sen2_val.append(sen_2_n[i])
#         label_val.append(label_n[i])
#     elif i >= val_num_n and i < test_num_n + val_num_n:
#         sen1_test.append(sen_1_n[i])
#         sen2_test.append(sen_2_n[i])
#         label_test.append(label_n[i])
#     elif i >=val_num_n + test_num_n:
#         sen1_train.append(sen_1_n[i])
#         sen2_train.append(sen_2_n[i])
#         label_train.append(label_n[i])
#
# with open('data/val/sens', 'w') as out_sen, open('data/val/label', 'w') as out_label:
#     for i in range(len(label_val)):
#         out_sen.write(str(i+1) + '\t' + sen1_val[i] + '\t' + sen2_val[i] + '\n')
#         out_label.write(str(i+1) + '\t' + label_val[i] + '\n')
#
# with open('data/test/sens', 'w') as out_sen, open('data/test/label', 'w') as out_label:
#     for i in range(len(label_test)):
#         out_sen.write(str(i+1) + '\t' + sen1_test[i] + '\t' + sen2_test[i] + '\n')
#         out_label.write(str(i+1) + '\t' + label_test[i] + '\n')
#
# with open('data/train/sens', 'w') as out_sen, open('data/train/label', 'w') as out_label:
#     for i in range(len(label_train)):
#         out_sen.write(str(i+1) + '\t' + sen1_train[i] + '\t' + sen2_train[i] + '\n')
#         out_label.write(str(i+1) + '\t' + label_train[i] + '\n')
# sens_1 = []
# sens_2 = []
# labels = []
# with open('data/temp_train/sens', 'r') as sen_in, open('data/temp_train/label', 'r') as label_in:
#     for line in label_in:
#         id, la = line.strip().split('\t')
#         labels.append(la)
#     for line in sen_in:
#         id, sen1, sen2 = line.strip().split('\t')
#         sens_1.append(sen1)
#         sens_2.append(sen2)
# p = 0
# n = 0
# for label in labels:
#     if label == '0':
#         n += 1
#     elif label == '1':
#         p += 1
# record = []
# max = len(labels)
# for i in range((n - p) / 2):
#     index = random.randint(0, max - 1)
#     while index in record:
#         index = random.randint(0, max - 1)
#     sens_1.append(sens_1[index])
#     sens_2.append(sens_1[index])
#     sens_1.append(sens_2[index])
#     sens_2.append(sens_2[index])
#     labels.append('1')
#     labels.append('1')
# with open('data/train/balabced_sen', 'w') as sen_out, open('data/train/balabced_label', 'w') as label_out:
#     for i in range(len(labels)):
#         sen_out.write(str(i + 1) + '\t' + sens_1[i] + '\t' +sens_2[i] + '\n')
#         label_out.write(str(i + 1) + '\t' + labels[i] + '\n')
sens_1 = []
sens_2 = []
labels = []
with open('data/temp_train/sens', 'r') as sens_in, open('data/temp_train/label', 'r') as label_in:
    for line in sens_in:
        id, sen1, sen2 = line.strip().split('\t')
        sens_1.append(sen1)
        sens_2.append(sen2)
    for line in label_in:
        id, label = line.strip().split('\t')
        labels.append(label)
negative = 0
positive = 0
pos_sen1 = []
pos_sen2 = []
neg_sen1 = []
neg_sen2 = []
for i in range(len(labels)):
    if labels[i] == '0':
        negative += 1
        neg_sen1.append(sens_1[i])
        neg_sen2.append(sens_2[i])
    if labels[i] == '1':
        positive += 1
        pos_sen1.append(sens_1[i])
        pos_sen2.append(sens_2[i])
M = 5
pos_num = len(pos_sen1)
neg_num = len(neg_sen1)
for i in range(M):
    neg_1 = []
    neg_2 = []
    neg_list = []
    for j in range(pos_num):
        index = random.randint(0, neg_num-1)
        while index in neg_list:
            index = random.randint(0, neg_num - 1)
        neg_list.append(index)
        neg_1.append(neg_sen1[index])
        neg_2.append(neg_sen2[index])
    with open('data/bag_train/sens_' + str(i + 1), 'w') as sens_out, open('data/bag_train/labels_' + str(i + 1), 'w') as labels_out:
        sens_temp = []
        labels_temp = []
        lineno = 0
        for j in range(pos_num):
            lineno += 1
            sens_temp.append(str(lineno) + '\t' + pos_sen1[j] + '\t' + pos_sen2[j] + '\n')
            labels_temp.append(str(lineno) + '\t' + '1' + '\n')
            lineno += 1
            sens_temp.append(str(lineno) + '\t' + neg_1[j] + '\t' + neg_2[j] + '\n')
            labels_temp.append(str(lineno) + '\t' + '0' + '\n')
        for line in sens_temp:
            sens_out.write(line)
        for line in labels_temp:
            labels_out.write(line)