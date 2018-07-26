import sys
import jieba
from collections import Counter
import matplotlib.pyplot as plt

# def analysis(inpath, labelpath, datapath):
# 	datas = []
# 	labels = []
# 	with open(inpath, 'r') as file:
# 		for line in file:
# 			id, sen_1, sen_2, label = line.strip().split('\t')
# 			labels.append(label+'\n')
# 			datas.append(id+'\t'+sen_1+'\t'+sen_2+'\n')
# 	with open(labelpath, 'w') as file:
# 		for label in labels:
# 			file.write(label)
# 	with open(datapath, 'w') as file:
# 		for data in datas:
# 			file.write(data)
#
# if __name__ == '__main__':
# 	analysis(sys.argv[1], sys.argv[2], sys.argv[3])
length = []
x = []
y = []
counter = Counter()
with open('data/train/train_add', 'r') as file1, open('data/test/test', 'r') as file2:
	for line in file1:
		id, sen1, sen2 = line.strip().split('\t')
		length.append(len([w for w in jieba.cut(sen1) if w.strip()]))
		length.append(len([w for w in jieba.cut(sen2) if w.strip()]))
	for line in file1:
		id, sen1, sen2 = line.strip().split('\t')
		length.append(len([w for w in jieba.cut(sen1) if w.strip()]))
		length.append(len([w for w in jieba.cut(sen2) if w.strip()]))
	dic = {}
	for num in length:
		if num in dic.keys():
			dic[num] +=1
		else:
			dic[num] = 1
	for num in dic.keys():
		x.append(num)
		y.append(dic[num])
	plt.plot(x, y)
	plt.show()

