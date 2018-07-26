import torch
from torch.autograd import Variable
import torch.nn as nn

class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, voc_size, voc_dim, num_layers=1):
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(voc_size, voc_dim)
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(p=0.2)


    def forward(self, input):
        embedded = self.embedding(input)
        hidden = self.init_hidden(input.size()[0])
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size, num_directions=1):
        return torch.ones(self.num_layers * num_directions, batch_size, self.hidden_size, dtype=torch.float)

class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        # self.w1 = nn.Linear(hidden_size, hidden_size)
        # self.w2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size , 2)
        self.softmax = nn.LogSoftmax(dim=0)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, hidden):
        out = self.out(hidden)
        out = self.softmax(out)
        return out