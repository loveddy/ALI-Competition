import torch
from torch.autograd import Variable
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, output_size, kernel_size, voc_size, voc_dim):
        super(CNNEncoder, self).__init__()
        self.channel = voc_dim
        self.embedding = nn.Embedding(voc_size, voc_dim)
        self.cnn = nn.Conv1d(voc_dim, output_size, kernel_size)
        self.drop = nn.Dropout(p=0.2)


    def forward(self, input):
        embedded = self.embedding(input)
        input_shape = embedded.size()
        output = self.cnn(embedded.view(input_shape[0], input_shape[2], input_shape[1]))
        return output



class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        # self.w1 = nn.Linear(hidden_size, hidden_size)
        # self.w2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size * 2, 2)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, hidden_1, hidden_2):
        out = self.out(torch.cat((hidden_1, hidden_2), dim=1))
        #out = self.sigmoid(out).unsqueeze(0)
        return out