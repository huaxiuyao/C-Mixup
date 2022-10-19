import pdb

import numpy as np
import os
import random
import shutil
import sys
import operator
from numbers import Number
from collections import OrderedDict
import argparse


import torch
from torch import nn
from torch.utils.data import Dataset


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


# Functions
def save_vars(vs, filepath):
    """
    Saves variables to the given filepath in a safe manner.
    """
    filepath = filepath
    if os.path.exists(filepath):
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)


def save_model(model, filepath):
    """
    To load a saved model, simply use
    `model.load_state_dict(torch.load('path-to-saved-model'))`.
    """
    save_vars(model.state_dict(), filepath)


def unpack_data(data, device):
    return data[0].to(device), data[1].to(device)


def save_best_model(model, runPath, agg, args):
    if agg['val_stat'][-1] > max(agg['val_stat'][:-1]):
        save_model(model, f'{runPath}/model.rar')
        save_vars(agg, f'{runPath}/losses.rar')
    elif agg['val_stat'][-1] == max(agg['val_stat'][:-1]) and agg['test_stat'][-1] > max(agg['test_stat'][:-1]):
        save_model(model, f'{runPath}/model.rar')
        save_vars(agg, f'{runPath}/losses.rar')


def single_class_predict_fn(yhat):
    _, predicted = torch.max(yhat.data, 1)

    return predicted


def return_predict_fn(dataset):
    return {
        'fmow': single_class_predict_fn,
        'camelyon': single_class_predict_fn,
        'poverty': lambda yhat: yhat,
        'iwildcam': single_class_predict_fn,
        'amazon': single_class_predict_fn,
        'civil': single_class_predict_fn,
        'cdsprites': single_class_predict_fn,
        'rxrx': single_class_predict_fn,
        'dataset': single_class_predict_fn,
        'cmnist': single_class_predict_fn,
        'celeba': single_class_predict_fn,
        'cub': single_class_predict_fn
    }[dataset]

def return_criterion(dataset):
    return {
        'fmow': nn.CrossEntropyLoss(),
        'camelyon': nn.CrossEntropyLoss(),
        'poverty': nn.MSELoss(),
        'iwildcam': nn.CrossEntropyLoss(),
        'amazon': nn.CrossEntropyLoss(),
        'civil': nn.CrossEntropyLoss(),
        'cdsprites': nn.CrossEntropyLoss(),
        'rxrx': nn.CrossEntropyLoss(),
        'dataset': nn.CrossEntropyLoss(),
        'cmnist': nn.CrossEntropyLoss(),
        'celeba': nn.CrossEntropyLoss(),
        'cub': nn.CrossEntropyLoss()
    }[dataset]


def save_pred(args, model, train_loader, epoch, save_dir, predict_fn, device):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    yhats, ys, idxes = [], [], []
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            model.eval()
            x, y, idx = data[0].to(device), data[1].to(device), data[-1].to(device)
            y_hat = model(x)
            ys.append(y.cpu())
            yhats.append(y_hat.cpu())
            idxes.append(idx.cpu())
            # if i > 10:
            #     break

        ypreds, ys, idxes = predict_fn(torch.cat(yhats)), torch.cat(ys), torch.cat(idxes)

        ypreds = ypreds[torch.argsort(idxes)]
        ys = ys[torch.argsort(idxes)]

        y = torch.cat([ys.reshape(-1, 1), ypreds.reshape(-1, 1)], dim=1)

        df = pd.DataFrame(y.cpu().numpy(), columns=['y_true', 'y_pred'])
        df.to_csv(os.path.join(save_dir, f"{args.dataset}_{args.algorithm}_{epoch}.csv"))

        # print accuracy
        wrong_labels = (df['y_true'].values == df['y_pred'].values).astype(int)
        from wilds.common.grouper import CombinatorialGrouper
        grouper = CombinatorialGrouper(train_loader.dataset.dataset.dataset, ['y', 'black'])
        group_array = grouper.metadata_to_group(train_loader.dataset.dataset.dataset.metadata_array).numpy()
        group_array = group_array[np.where(
            train_loader.dataset.dataset.dataset.split_array == train_loader.dataset.dataset.dataset.split_dict[
                'train'])]
        for i in np.unique(group_array):
            idxes = np.where(group_array == i)[0]
            print(f"domain = {i}, length = {len(idxes)}, acc = {np.sum(wrong_labels[idxes] / len(idxes))} ")

        def print_group_info(idxes):
            group_ids, group_counts = np.unique(group_array[idxes], return_counts=True)
            for idx, j in enumerate(group_ids):
                print(f"group[{j}]: {group_counts[idx]} ")

        correct_idxes = np.where(wrong_labels == 1)[0]
        print("correct points:")
        print_group_info(correct_idxes)
        wrong_idxes = np.where(wrong_labels == 0)[0]
        print("wrong points:")
        print_group_info(wrong_idxes)
