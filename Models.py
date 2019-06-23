import torch
import torch.nn as nn
import torch.nn.functional as F

class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        # print("x", x[0])
        # print("y", y[0])
        # print("z", z[0])
        # print(self.embeddingnet)
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        # print("embedded_x", embedded_x[0])
        # print("embedded_y", embedded_y[0])
        # print("embedded_z", embedded_z[0])
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z

class Test_net(nn.Module):
    def __init__(self, embeddingnet):
        super(Test_net, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x):
        # print("x", x)
        embedded_x = self.embeddingnet(x)
        # print("embedded_x", embedded_x)
        return embedded_x
# Triplet_testなんてやるより
# for i in input:
#     result.append(embeddingnet(i))
# とかにすればいいけど
class N_PAIR_net(nn.Module):
    def __init__(self, embeddingnet):
        super(N_PAIR_net, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, anchors, positives):
        f = self.embeddingnet(anchors)
        f_p = self.embeddingnet(positives)
        return f, f_p

# ほぼTripletnetと同じだけど一応わけとく
class N_plus_1_net(nn.Module):
    def __init__(self, embeddingnet):
        super(N_plus_1_net, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, anchor, positive, negatives):
        embedded_anchor = self.embeddingnet(anchor)
        embedded_positive = self.embeddingnet(positive)
        embedded_negatives = self.embeddingnet(negatives)
        return embedded_anchor, embedded_positive, embedded_negatives
