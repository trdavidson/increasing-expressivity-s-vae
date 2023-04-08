# Utilities to perform classification given embeddings
# currently K-NN
import torch
import numpy as np


def euclidean_dist(a, b):
    """
    Euclidean Distance
    :param a: labeled embeddings, [labels, 1, dim]
    :param b: to_label embeddings, [1, to_label, dim]
    :return: distance to_label to all labeled
    """
    return ((a - b) ** 2).sum(-1)


# https://en.wikipedia.org/wiki/Cosine_similarity
def angular_dist(a, b):
    """
    Angular distance
    :param a: labeled embeddings, [labels, 1, dim]
    :param b: to_label embeddings, [1, to_label, dim]
    :return: distance to_label to all labeled [labels, to_label]
    """
    norm = torch.norm(a, dim=2) * torch.norm(b, dim=2)  # allow for mixed norm elements
    cos_sim = (a * b).sum(-1) / norm
    cos_sim = torch.clamp(cos_sim, min=1e-6, max=1-1e-6)  # clamp to not allow 0. and 1.

    return torch.acos(cos_sim) / np.pi


def classify(a, b, al, bl, k=1, metric=angular_dist, v=0):
    """
    K-Nearest Neighbors classifier
    :param a: labeled embeddings, [labels, dim]
    :param b: to_label embeddings, [to_label, dim]
    :param al: labels [labels]
    :param bl: to_label [to_label]
    :param k: k-nearest neighbors
    :param metric: metric to calculate distance with
    :return: accuracy
    """
    # reshape labeled and to-label for correct permutation
    a = a.reshape(a.shape[0], 1, a.shape[1])  # [a, 1, dim]
    b = b.reshape(1, b.shape[0], b.shape[1])  # [1, b, dim]
    # get distances of all points to labeled points
    d = metric(a, b)  # [labels, to-label]
    # get idx of top k nearest points
    idx = d.topk(k=k, dim=0, largest=False)[1]  # [k, b]
    # sample uniform indices
    u = np.random.randint(0, k, bl.shape[0])  # [b]
    # pick k-nn prediction
    predictions = al[idx][u, np.arange(al[idx].shape[1])]  # [b]
    # get prediction accuracy
    acc = (bl == predictions).float().mean()  # scalar

    if v > 0:
        print('num labels: %d \t k-nn: %d \t acc: %.4f' % (al.shape[0], k, acc))

    return acc
