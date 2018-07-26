import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, kernel_size, voc_size, voc_dim):
        super(Model, self).__init__()
        self.channel = voc_dim
        self.embedding = nn.Embedding(voc_size, voc_dim)
        self.cnn = nn.Sequential(
            nn.Tanh(),
            # nn.Dropout(0.6),
            nn.Conv2d(1, 100, 4, bias=False),
            nn.Tanh(),
            nn.MaxPool2d(3, 3),
            # nn.Dropout(0.6),
            nn.Conv2d(100, 100, 3, bias=False),
            nn.Tanh(),
            nn.MaxPool2d((3, 2), 2)
            # nn.Dropout(0.6),
        )
        self.linear = nn.Sequential(
            nn.Linear(48, 24),
            nn.Tanh(),
            nn.Linear(24, 1),
            nn.Tanh()

        )
        self.drop = nn.Dropout(p=0.5)
        self.norm = nn.BatchNorm1d(50)

    def forward(self, input_1, input_2):
        embedded_1 = self.embedding(input_1)
        embedded_2 = self.embedding(input_2)
        input_shape = embedded_1.size()
        cnn_output_1 = self.linear(self.cnn(embedded_1.view(input_shape[0], 1, input_shape[2], input_shape[1])).view(input_shape[0], 100, 48))
        cnn_output_2 = self.linear(self.cnn(embedded_2.view(input_shape[0], 1, input_shape[2], input_shape[1])).view(input_shape[0], 100, 48))

        score = torch.norm(cnn_output_1 - cnn_output_2, 2, dim=1)

        # output = torch.cat((1. - score, score), dim=1)
        return score.view(input_shape[0])

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.cnn[1].weight)
        torch.nn.init.xavier_uniform_(self.cnn[4].weight)

# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         self.distance = nn.CosineSimilarity(dim=1)
#         self.classifier_value = nn.Parameter(torch.tensor(data=0.5, dtype=torch.float))
#
#     def forward(self, hidden_1, hidden_2):
#         sign = self.distance(hidden_1, hidden_2) - self.classifier_value
#
#         return out.view(-1, 1)

# class ContrastiveLoss(torch.nn.Module):
#     """
#     Contrastive loss function.
#     Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     """
#
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, output1, output2, label):
#         euclidean_distance = F.pairwise_distance(output1.view(1,-1), output2.view(1,-1))
#         loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
#                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#
#
#         return loss_contrastive
