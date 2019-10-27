import os
import numpy as np
import torch
import logging
from collections import namedtuple

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)

def to_tuple(x):
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
        return (x, x)

    raise ValueError('Value should be tuple of length 2 or int value, got "{}"'.format(x))

def distance_star(x,y,grad_y):
    max_grad_cord = grad_y.abs().view(grad_y.shape[0], -1).argmax(dim=-1)
    return (x.view(x.shape[0], -1)[:,max_grad_cord].diagonal() - y.view(y.shape[0], -1)[:,max_grad_cord].diagonal()).abs()


def get_cuda_ifavailable(torch_obj):
    if (torch.cuda.is_available()):
        return torch_obj.cuda()
    else:
        return torch_obj

def configure_logging(debug,quiet,logfile):
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    elif quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("Controlling Neural Levelsets - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if logfile is not None:
        file_logger_handler = logging.FileHandler(logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def get_dist_matrix(x,y):
    x_square = (x ** 2).sum(dim=-1, keepdim=True)
    y_square = (y ** 2).sum(dim=-1).unsqueeze(0)
    zz = torch.mm(x, y.transpose(1, 0))
    P = x_square + y_square - 2 * zz
    return P

def dict_to_nametuple(name,d):
    return namedtuple(name, d.keys())(*d.values())

def fps( points, B):

    r = np.sum(points * points, 1)
    r = np.expand_dims(r, axis=1)
    distance = r - 2 * np.matmul(points, np.transpose(points, [1, 0])) + np.transpose(r, [1, 0])

    def getGreedyPerm(D,B):
        """
        A Naive O(N^2) algorithm to do furthest points sampling

        Parameters
        ----------
        D : ndarray (N, N)
            An NxN distance matrix for points
        Return
        ------
        tuple (list, list)
            (permutation (N-length array of indices),
            lambdas (N-length array of insertion radii))
        """

        # By default, takes the first point in the list to be the
        # first point in the permutation, but could be random
        perm = np.zeros(B, dtype=np.int64)
        lambdas = np.zeros(B)
        perm[0] = np.random.choice(np.arange(D.shape[0]),1).item()
        ds = D[perm[0], :]
        for i in range(1, B):
            idx = np.argmax(ds)
            perm[i] = idx
            lambdas[i] = ds[idx]
            ds = np.minimum(ds, D[idx, :])
        return (perm, lambdas)

    idx,_ = getGreedyPerm(distance.squeeze(),B)
    return idx

