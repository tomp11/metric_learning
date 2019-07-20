import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Angular_mc_loss(nn.Module):
    def __init__(self, alpha=45, in_degree=True):
        super(Angular_mc_loss, self).__init__()
        if in_degree:
            alpha = np.deg2rad(alpha)
        self.sq_tan_alpha = np.tan(alpha) ** 2

    def forward(self, embeddings, target, with_npair=True, lamb=2):
        n_pairs = self.get_n_pairs(target)
        n_pairs = n_pairs.cuda()
        f = embeddings[n_pairs[:, 0]]
        f_p = embeddings[n_pairs[:, 1]]
        # print(f, f_p)
        term1 = 4 * self.sq_tan_alpha * torch.matmul(f + f_p, torch.transpose(f_p, 0, 1))
        term2 = 2 * (1 + self.sq_tan_alpha) * torch.sum(f * f_p, keepdim=True, dim=1)
        f_apn = term1 - term2
        mask = torch.ones_like(f_apn) - torch.eye(len(f)).cuda()
        f_apn = f_apn * mask
        loss = torch.mean(torch.logsumexp(f_apn, dim=1))
        if with_npair:
            loss_npair = self.n_pair_mc_loss(f, f_p)
            # print(loss, loss_npair)
            loss = loss_npair + lamb*loss
        # Preventing overflow
        # with torch.no_grad():
        #     t = torch.max(x, dim=2)[0] # (batch_size, 1)
        # print(t.size())
        #
        # x = torch.exp(x - t.unsqueeze(dim=1))
        # x = torch.log(torch.exp(-t) + torch.sum(x, 2))
        # loss = torch.mean(t + x)
        return loss

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
        """
        labels = labels.cpu().data.numpy()
        n_pairs = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = np.random.choice(label_indices, 2, replace=False)
            n_pairs.append([anchor, positive])
        n_pairs = np.array(n_pairs)
        return torch.LongTensor(n_pairs)

    @staticmethod
    def n_pair_mc_loss(f, f_p):
        n_pairs = len(f)
        term1 = torch.matmul(f, torch.transpose(f_p, 0, 1))
        term2 = torch.sum(f * f_p, keepdim=True, dim=1)
        f_apn = term1 - term2
        mask = torch.ones_like(f_apn) - torch.eye(n_pairs).cuda()
        f_apn = f_apn * mask
        return torch.mean(torch.logsumexp(f_apn, dim=1))

class n_pair_mc_loss(nn.Module):
    def __init__(self):
        super(n_pair_mc_loss, self).__init__()

    def forward(self, f, f_p):
        n_pairs = len(f)
        term1 = torch.matmul(f, torch.transpose(f_p, 0, 1))
        term2 = torch.sum(f * f_p, keepdim=True, dim=1)
        f_apn = term1 - term2
        mask = torch.ones_like(f_apn) - torch.eye(n_pairs).cuda()
        f_apn = f_apn * mask
        return torch.mean(torch.logsumexp(f_apn, dim=1))


class N_plus_1_angularLoss(nn.Module):
    """
    Angular loss
    Wang, Jian. "Deep Metric Learning with Angular Loss," CVPR, 2017
    https://arxiv.org/pdf/1708.01682.pdf
    """

    def __init__(self, l2_reg=0.02, angle_bound=1., lambda_ang=2):
        super(my_AngularLoss, self).__init__()
        self.l2_reg = l2_reg
        self.angle_bound = angle_bound
        self.lambda_ang = lambda_ang
        self.softplus = nn.Softplus()

    def forward(self, anchors, positives, negatives):

        losses = self.angular_loss(anchors, positives, negatives, self.angle_bound) + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    @staticmethod
    def angular_loss(anchors, positives, negatives, angle_bound=1.):
        """
        Calculates angular loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :param angle_bound: tan^2 angle
        :return: A scalar
        """

        anchors = torch.unsqueeze(anchors, dim=1) # (batch_size, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1) # (batch_size, 1, embedding_size)
        batch_size = anchors.size()[0]
        negatives = [negatives[i*5:(i+1)*5] for i in range(batch_size)]
        negatives = torch.stack(negatives)# (batch_size, n-1, embedding_size)

        anchors, positives, negatives = anchors.cuda(), positives.cuda(), negatives.cuda()

        x = 4. * angle_bound * torch.matmul((anchors + positives), negatives.transpose(1, 2)) - 2. * (1. + angle_bound) * torch.matmul(anchors, positives.transpose(1, 2))  # (n, 1, n-1)

        print(x.size())
        # Preventing overflow
        with torch.no_grad():
            t = torch.max(x, dim=2)[0] # (batch_size, 1)
        print(t.size())

        x = torch.exp(x - t.unsqueeze(dim=1))
        x = torch.log(torch.exp(-t) + torch.sum(x, 2))
        loss = torch.mean(t + x)

        return loss

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]



class N_plus_1_Loss(nn.Module):
    """
    N-Pair loss
    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
    """

    def __init__(self, l2_reg=0.02):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchors, positives, negatives):
        """
        anchors (batch_size, embedding_size)
        positives (batch_size, embedding_size)
        negatives (batch_size*(n-1), embedding_size)
        """
        batch_size = anchors.size()[0]
        negatives = [negatives[i*5:(i+1)*5] for i in range(batch_size)]
        negatives = torch.stack(negatives)# (batch_size, n-1, embedding_size)

        # print(anchors)
        anchors, positives, negatives = anchors.cuda(), positives.cuda(), negatives.cuda()
        losses = self.n_pair_loss(anchors, positives, negatives) \
            + self.l2_reg * self.l2_loss(anchors, positives)
        # print(self.n_pair_loss(anchors, positives, negatives), self.l2_reg * self.l2_loss(anchors, positives))
        return losses


    @staticmethod
    def n_pair_loss(anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1+x))
        return loss

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]
