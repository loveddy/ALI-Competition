import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()



    def forward(self):


    def init_parameters(self):

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
