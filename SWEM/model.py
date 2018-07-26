import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,max_length, voc_size, voc_dim):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(voc_size, voc_dim)
        self.max_pool = nn.MaxPool1d(max_length, max_length)
        self.drop = nn.Dropout(p=0.8)
        self.mlp = nn.Sequential(
            nn.Linear(600, 600),
            nn.LeakyReLU(10),
            nn.Linear(600, 300),
            nn.LeakyReLU(10),
            nn.Linear(300, 300),
            nn.LeakyReLU(10),
            nn.Linear(300, 100),
            nn.LeakyReLU(10),
            nn.Linear(100, 2)
        )


    def forward(self, input_1, input_2):
        embedded_1 = self.embedding(input_1)
        embedded_2 = self.embedding(input_2)
        input_shape = embedded_1.size()
        max_1 = self.max_pool(embedded_1.view(input_shape[0], input_shape[2], input_shape[1])).view(input_shape[0], -1)
        max_2 = self.max_pool(embedded_1.view(input_shape[0], input_shape[2], input_shape[1])).view(input_shape[0], -1)
        mean_1 = torch.mean(embedded_1, dim=1).view(input_shape[0], -1)
        mean_2 = torch.mean(embedded_2, dim=1).view(input_shape[0], -1)
        feature_1 = torch.cat((max_1, mean_1), dim=1)
        feature_2 = torch.cat((max_2, mean_2), dim=1)
        output = self.mlp(feature_1 - feature_2)
        return output

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        torch.nn.init.xavier_uniform_(self.mlp[4].weight)
        torch.nn.init.xavier_uniform_(self.mlp[6].weight)
        torch.nn.init.xavier_uniform_(self.mlp[8].weight)

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
