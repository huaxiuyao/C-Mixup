import torch
import numpy as np
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

def stats_values(targets, flag = False):
    mean = torch.mean(targets)
    min = torch.min(targets)
    max = torch.max(targets)
    std = torch.std(targets)
    if flag:
        print(f'y stats: mean = {mean}, max = {max}, min = {min}, std = {std}')
    return mean, min, max, std


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize = 2, device = "cuda"):
        self.cuda = cuda
        self.P = window # window -> model input
        self.h = horizon # 
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin,delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape # m -> dim 2, n
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train+valid) * self.n), self.n)
        
        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)
            
        if self.cuda:
            self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)
        
        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))
    
    def _normalized(self, normalize):
        #normalized by the maximum value of entire matrix.
       
        if (normalize == 0):
            self.dat = self.rawdat
            
        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)
            
        #normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:,i]))
                self.dat[:,i] = self.rawdat[:,i] / np.max(np.abs(self.rawdat[:,i]))
            
        
    def _split(self, train, valid, test):
        
        train_set = range(self.P+self.h-1, train) # idx on dim 0
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)
        
        
    def _batchify(self, idx_set, horizon):
        
        n = len(idx_set) # n -> dim 0
        X = torch.zeros((n,self.P,self.m)) # x size
        Y = torch.zeros((n,self.m))
        
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P # end - window ----> end, end = idx - horizon + 1[predict len]
            X[i,:,:] = torch.from_numpy(self.dat[start:end, :]) # window len
            Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :])

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True, device = "cuda"):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            if self.cuda:
                X = X.to(device)
                Y = Y.to(device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

def get_TimeSeries_data_packet(args, ts_data):
    data_packet = {
        'x_train': ts_data.train[0],
        'x_valid': ts_data.valid[0],
        'x_test': ts_data.test[0],
        'y_train': ts_data.train[1],
        'y_valid': ts_data.valid[1],
        'y_test': ts_data.test[1],
    }
    return data_packet