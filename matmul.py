import torch

# a = torch.randn(64, 256)
# p = torch.randn(64, 256)
# n = torch.randn(64, 63, 256)
def angular_loss(anchors, positives, negatives, angle_bound=1.):
        """
        Calculates angular loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :param angle_bound: tan^2 angle
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = 4. * angle_bound * torch.matmul((anchors + positives), negatives.transpose(1, 2)) \
            - 2. * (1. + angle_bound) * torch.matmul(anchors, positives.transpose(1, 2))  # (n, 1, n-1)
        y = torch.matmul((anchors + positives), negatives.transpose(1, 2))
        print(y.size())
        # Preventing overflow
        with torch.no_grad():
            t = torch.max(x, dim=2)[0]

        x = torch.exp(x - t.unsqueeze(dim=1))
        x = torch.log(torch.exp(-t) + torch.sum(x, 2))
        loss = torch.mean(t + x)
        # print(x, t)
        return loss

# loss = angular_loss(a, p, n)
# print(loss)


paths = []
for line in open("n_plus_1_index_text.txt"):
    paths.append((line.split()[0], line.split()[1], line.split()[2:]))
print(paths)
