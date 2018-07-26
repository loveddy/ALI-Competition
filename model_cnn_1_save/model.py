import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, output_size, kernel_size, voc_size, voc_dim):
        super(CNNEncoder, self).__init__()
        self.channel = voc_dim
        self.embedding = nn.Embedding(voc_size, voc_dim)
        self.cnn = nn.Sequential(nn.Conv1d(voc_dim, 64, kernel_size),
                                 nn.ReLU(),
                                 nn.MaxPool1d(2, 2),
                                 nn.Conv1d(64, 32, kernel_size - 1),
                                 nn.ReLU(),
                                 nn.MaxPool1d(2, 2)
                                 )
        self.out = nn.Sequential(nn.Linear(32, 32),
                                 nn.Linear(32, 16)
                                 )

    def forward(self, input):
        embedded = self.embedding(input)
        input_shape = embedded.size()
        cnn_output = self.cnn(embedded.view(input_shape[0], input_shape[2], input_shape[1]))
        output = self.out(torch.mean(cnn_output, dim=2))
        return output


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.out = nn.Linear(1, 2)

    def forward(self, hidden_1, hidden_2):
        norm_1d_1 = torch.norm(hidden_1, dim=1, p=1, keepdim=True)
        norm_1d_2 = torch.norm(hidden_2, dim=1, p=1, keepdim=True)
        distance = torch.exp(torch.norm(norm_1d_1 - norm_1d_2, dim=1, p=1, keepdim=True) * -1.)
        return distance
