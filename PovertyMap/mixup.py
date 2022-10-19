import pdb

import torch
import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mix_up(args, x, y, x2=None, y2=None, dataset = 'none'):

    # y1, y2 should be one-hot label, which means the shape of y1 and y2 should be [bsz, n_classes]

    if x2 is None:
        idxes = torch.randperm(len(x))
        x1 = x
        x2 = x[idxes]
        y1 = y
        y2 = y[idxes]
    else:
        x1 = x
        y1 = y

    n_classes = 1 if dataset == 'poverty' else y1.shape[1]
    bsz = len(x1)
    l = np.random.beta(args.mix_alpha, args.mix_alpha, [bsz, 1])
    if len(x1.shape) == 4:
        l_x = np.tile(l[..., None, None], (1, *x1.shape[1:]))
    else:
        l_x = np.tile(l, (1, *x1.shape[1:]))
    if dataset != 'poverty':
        l_y = np.tile(l, [1, n_classes])
    else:
        l_y = l
        y1 = y1.reshape(-1,1)
        y2 = y2.reshape(-1,1)

    # mixed_input = l * x + (1 - l) * x2
    mixed_x = torch.tensor(l_x, dtype=torch.float32).to(x1.device) * x1 + torch.tensor(1-l_x, dtype=torch.float32).to(x2.device) * x2
    mixed_y = torch.tensor(l_y, dtype=torch.float32).to(y1.device) * y1 + torch.tensor(1-l_y, dtype=torch.float32).to(y2.device) * y2

    return mixed_x, mixed_y

