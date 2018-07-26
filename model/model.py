import torch
import torch.nn.functional as F
import torch.nn as nn

class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, voc_size, voc_dim, num_layers=1):
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(voc_size, voc_dim)
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(p=0.4)


    def forward(self, input, lengths=None):
        hidden, cell = self.init_hidden(input.size()[0])
        embedded = self.embedding(input)
        if not lengths == None:
            embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, True)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        if not lengths == None:
            output = torch.nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
        return output, hidden, cell

    def init_hidden(self, batch_size, num_directions=1):
        return torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size), torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size)

class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.out = nn.Linear(hidden_size * 2, 2)

    def forward(self, hidden_1, hidden_2):
        input = torch.cat((hidden_1.view(1, -1), hidden_2.view(1, -1)), dim=1)
        out = self.out(input)
        return out
        # norm_1d_1 = torch.norm(hidden_1.view(1,-1), dim=1, p=1, keepdim=True)
        # norm_1d_2 = torch.norm(hidden_2.view(1,-1), dim=1, p=1, keepdim=True)
        # distance = torch.exp(torch.norm(norm_1d_1 - norm_1d_2, dim=1, p=1, keepdim=True) * -1.)
        # return distance

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1.view(1,-1), output2.view(1,-1))
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive