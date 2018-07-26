import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, output_size, kernel_size, voc_size, voc_dim):
        super(CNNEncoder, self).__init__()
        self.channel = voc_dim
        self.embedding = nn.Embedding(voc_size, voc_dim)
        self.cnn_1 = nn.Sequential(nn.Conv1d(voc_dim, 128, kernel_size + 1),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2, 2),
                                   )
        self.cnn_2 = nn.Sequential(nn.Conv1d(voc_dim, 64, kernel_size),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2, 2),
                                   )
        self.cnn_3 = nn.Sequential(nn.Conv1d(voc_dim, 64, kernel_size-1),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2, 2),
                                   )
        self.out = nn.Sequential(nn.Linear(256, 128),
                                 nn.Linear(128, 64),
                                 nn.Linear(64, 32)
                                 )

    def forward(self, input):
        embedded = self.embedding(input)
        input_shape = embedded.size()
        cnn_1 = torch.mean(self.cnn_1(embedded.view(input_shape[0], input_shape[2], input_shape[1])), dim=2)
        cnn_2 = torch.mean(self.cnn_2(embedded.view(input_shape[0], input_shape[2], input_shape[1])), dim=2)
        cnn_3 = torch.mean(self.cnn_3(embedded.view(input_shape[0], input_shape[2], input_shape[1])), dim=2)
        output = self.out(torch.cat((cnn_1, cnn_2, cnn_3), dim=1))
        return output


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

    def forward(self, hidden_1, hidden_2):
        norm_1d_1 = torch.norm(hidden_1, dim=1, p=2, keepdim=True)
        norm_1d_2 = torch.norm(hidden_2, dim=1, p=2, keepdim=True)
        distance = torch.exp(torch.norm(norm_1d_1 - norm_1d_2, dim=1, p=1, keepdim=True) * -1.)
        return distance
