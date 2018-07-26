#coding=utf8
import sys
import os

def evaluate(result_path, tar):
    result = []
    target = []
    module_path = os.path.dirname(__file__)
    with open(module_path+result_path, 'r') as file:
        for line in file:
            id, label = line.strip().split('\t')
            result.append(label)

    with open(module_path+tar, 'r') as file:
        for line in file:
            id, label = line.strip().split('\t')
            target.append(label)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(target)):
        if result[i] == "1" and target[i] == "1":
            TP += 1
        if result[i] == "1" and target[i] == "0":
            FP += 1
        if result[i] == "0" and target[i] == "0":
            TN += 1
        if result[i] == "0" and target[i] == "1":
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
    print " precision   recall   accuracy   f1_score"
    print "    %.2f       %.2f      %.2f       %.2f " % (precision, recall, accuracy, f1_score)

if __name__ == '__main__':
    evaluate(sys.argv[1], sys.argv[2])