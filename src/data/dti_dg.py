 # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Adapted by TDC.

import os
from random import shuffle
import torch
import numpy as np
import pandas as pd
import pickle
import PIL
import json


from .Dti_dg_lib import datasets
from .Dti_dg_lib import hparams_registry
#from .Dti_dg import algorithms
from .Dti_dg_lib.lib import misc
from .Dti_dg_lib.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

# ---> https://github.com/mims-harvard/TDC/tree/master/

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)

def stats_values(targets, flag = False):
    mean = torch.mean(targets)
    min = torch.min(targets)
    max = torch.max(targets)
    std = torch.std(targets)
    if flag:
        print(f'y stats: mean = {mean}, max = {max}, min = {min}, std = {std}')
    return mean, min, max, std

def add_noise(targets, noise_ratio = 0.1, flag = False):
    if flag:
        print(f'targets.shape = {targets.shape}')
        print(f'target[0,:20] = {targets[0,:20]}')
    mean, min, max, std = stats_values(targets,flag)
    noise_vec = torch.FloatTensor(np.random.randn(targets.shape[0])) * noise_ratio * std
    noise_vec = noise_vec.unsqueeze(1)
    if flag:
        
        print(f'targets.shape = {targets.shape}, noise_vec.shape = {noise_vec.shape}')
    stats_values(noise_vec,flag)
    targets += noise_vec
    return targets

class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 2            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
       'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

MAX_SEQ_PROTEIN = 1000
MAX_SEQ_DRUG = 100

from sklearn.preprocessing import OneHotEncoder
enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))

def protein_2_embed(x):
	return enc_protein.transform(np.array(x).reshape(-1,1)).toarray().T

def drug_2_embed(x):
	return enc_drug.transform(np.array(x).reshape(-1,1)).toarray().T

def trans_protein(x):
	temp = list(x.upper())
	temp = [i if i in amino_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_PROTEIN:
		temp = temp + ['?'] * (MAX_SEQ_PROTEIN-len(temp))
	else:
		temp = temp [:MAX_SEQ_PROTEIN]
	return temp

def trans_drug(x):
	temp = list(x)
	temp = [i if i in smiles_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_DRUG:
		temp = temp + ['?'] * (MAX_SEQ_DRUG-len(temp))
	else:
		temp = temp [:MAX_SEQ_DRUG]
	return temp

from torch.utils import data

class dti_tensor_dataset(data.Dataset):

    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        d = self.df.iloc[index].Drug_Enc
        t = self.df.iloc[index].Target_Enc
        
        d = drug_2_embed(d)
        t = protein_2_embed(t)
        
        y = self.df.iloc[index].Y

        print(f'y.shape = {y.shape}')
        return d, t, y

class TdcDtiDg(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        
        ENVIRONMENTS = [str(i) for i in list(range(2013, 2022))]
        TRAIN_ENV = [str(i) for i in list(range(2013, 2019))]
        TEST_ENV = ['2019', '2020', '2021']
        
        self.input_shape = [(63, 100), (26, 1000)] # drug and protein # 6300 and 26000
        self.num_classes = 1
        
        if root is None:
            raise ValueError('Data directory not specified!')
       
        ## create a datasets object
        self.datasets = []
        from tdc import BenchmarkGroup
        self.group = BenchmarkGroup(name = 'DTI_DG_Group', path = root)
        benchmark = self.group.get('BindingDB_Patent') 
        train_val, test, name = benchmark['train_val'], benchmark['test'], benchmark['name']
        
        unique_drug = pd.Series(train_val['Drug'].unique()).apply(trans_drug)
        print(f'unique_drug.shape= {unique_drug.shape}')

        unique_dict_drug = dict(zip(train_val['Drug'].unique(), unique_drug))

        train_val['Drug_Enc'] = [unique_dict_drug[i] for i in train_val['Drug']]

        unique_target = pd.Series(train_val['Target'].unique()).apply(trans_protein)
        unique_dict_target = dict(zip(train_val['Target'].unique(), unique_target))
        train_val['Target_Enc'] = [unique_dict_target[i] for i in train_val['Target']]

        for i in TRAIN_ENV:
            df_data = train_val[train_val.Year == int(i)]
            self.datasets.append(dti_tensor_dataset(df_data))
            print('Year ' + i + ' loaded...')

        unique_drug = pd.Series(test['Drug'].unique()).apply(trans_drug)
        unique_dict_drug = dict(zip(test['Drug'].unique(), unique_drug))
        test['Drug_Enc'] = [unique_dict_drug[i] for i in test['Drug']]

        unique_target = pd.Series(test['Target'].unique()).apply(trans_protein)
        unique_dict_target = dict(zip(test['Target'].unique(), unique_target))
        test['Target_Enc'] = [unique_dict_target[i] for i in test['Target']]

        for i in TEST_ENV:
            df_data = test[test.Year == int(i)]
            self.datasets.append(dti_tensor_dataset(df_data))
            print('Year ' + i + ' loaded...')

def get_all_batches(loaders,len_list,name):
    all_batches = []
    assay_len = []

    for i in range(len(loaders)):
        tl = zip(*([loaders[i]]))
        assay_num = 0
        steplen = int(len_list[i]) 
        for step in range(steplen):
            # if step == steplen - 1:
            #     print('For {} dataloader: {}, steplen = {}, packet : {:.3f}%, assay_num = {}'.format(name,i,steplen,step*100/(steplen),assay_num))
            if name == 'test' and i == 2: # 2021
                batch_list = [(d.float(), t.float(), y.float()) for d,t,y in loaders[i]]
            else:
                batch_list = [(d.float(), t.float(), y.float()) for d,t,y in next(tl)]             
            batch_len_all = sum([batch_list[i][0].shape[0] for i in range(len(batch_list))])
            assay_num += batch_len_all
            all_batches = all_batches + batch_list
        
        assay_len.append(assay_num)        
    return all_batches,assay_len

DTI_file_name = "DTI_sub100" #"DTI_sub100.pkl"
####### modifed from TDC #######
def get_data_packet(args, train_loaders,in_len,val_loaders,out_len,eval_loaders,test_len):
    if args.read_dataset:
        data_packet = read_dti_sub_dataset(args)
    else:
        print('Begin construct dti_dg data_packet')
        data_packet = {}

        ##### train #####
        train_all_batches,train_all_assay_len = get_all_batches(train_loaders,in_len,'train')
        tmp1 = torch.cat([sub[0] for sub in train_all_batches])
        tmp2 = torch.cat([sub[1] for sub in train_all_batches])
        tmp1 = tmp1.reshape(tmp1.shape[0],-1)
        tmp2 = tmp2.reshape(tmp2.shape[0],-1)

        print(f'in train, tmp1.shape = {tmp1.shape}, tmp2.shape = {tmp2.shape}')
        print(f'train_all_assay_len = {train_all_assay_len}, sum(train_all_assay_len) = {sum(train_all_assay_len)}')

        shuffle_idx = np.random.permutation(np.arange(tmp1.shape[0]))
        data_packet['x_train'] = torch.cat([tmp1,tmp2],dim=1)[shuffle_idx]

        data_packet['y_train'] = torch.cat([sub[2] for sub in train_all_batches]).unsqueeze(1)[shuffle_idx]

        data_packet['idx2assay_train'] = list(torch.tensor([i for i in range(len(train_all_assay_len)) for _ in range(train_all_assay_len[i])])[shuffle_idx])

        print(f"len(data_packet['idx2assay_train']) = {len(data_packet['idx2assay_train'])}")
        
        # data_packet['assay2idx_train'] = {loc:torch.nonzero(torch.tensor(data_packet['idx2assay_train'] == loc)).squeeze(-1)
                                        # for loc in np.unique(data_packet['idx2assay_train'])}

        ###### valid ######
        valid_all_batches, valid_all_assay_len = get_all_batches(val_loaders,out_len,'valid')
        tmp1 = torch.cat([sub[0] for sub in valid_all_batches])
        tmp2 = torch.cat([sub[1] for sub in valid_all_batches])
        tmp1 = tmp1.reshape(tmp1.shape[0],-1)
        tmp2 = tmp2.reshape(tmp2.shape[0],-1)
        shuffle_idx = np.random.permutation(np.arange(tmp1.shape[0]))
        data_packet['x_valid'] = torch.cat([tmp1,tmp2],dim=1)[shuffle_idx]

        data_packet['y_valid'] = torch.cat([sub[2] for sub in valid_all_batches]).unsqueeze(1)[shuffle_idx]

        ###### test ######
        test_all_batches,test_all_assay_len = get_all_batches(eval_loaders[0:2],test_len[0:2],'test') # 2021 is too less
        tmp1 = torch.cat([sub[0] for sub in test_all_batches])
        tmp2 = torch.cat([sub[1] for sub in test_all_batches])
        tmp1 = tmp1.reshape(tmp1.shape[0],-1)
        tmp2 = tmp2.reshape(tmp2.shape[0],-1)
        shuffle_idx = np.random.permutation(np.arange(tmp1.shape[0]))
        data_packet['x_test'] = torch.cat([tmp1,tmp2],dim=1)[shuffle_idx]
        data_packet['y_test'] = torch.cat([sub[2] for sub in test_all_batches]).unsqueeze(1)[shuffle_idx]

        #for debug test
        idx2assay_test = list(torch.tensor([i for i in range(len(test_all_assay_len)) for _ in range(test_all_assay_len[i])])[shuffle_idx])
        assay2idx_test_list = [torch.nonzero(torch.tensor(idx2assay_test == loc)).squeeze(-1)
                                        for loc in np.unique(idx2assay_test)]
        data_packet['x_test_assay_list'] = [data_packet['x_test'][tmpidx] for tmpidx in assay2idx_test_list]
        data_packet['y_test_assay_list'] = [data_packet['y_test'][tmpidx] for tmpidx in assay2idx_test_list]
        # data_packet['idx2assay_test'] = idx2assay_test

        store_dti_sub_dataset(args, data_packet)

    return data_packet

def store_x_y(args, x, y, name):
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    # store
    join_name = f"{DTI_file_name}_{name}"
    pickle.dump((x, y), open(os.path.join(args.data_dir, f"{join_name}.pkl"), 'wb'))
    print(f'Construct {join_name} to {args.data_dir}')

def read_x_y(args, name):
    join_name = f"{DTI_file_name}_{name}"
    x, y = pickle.load(open(os.path.join(args.data_dir, f"{join_name}.pkl"), 'rb'))
    print(f'Read {join_name} from {args.data_dir}')
    return x, y

def store_dti_sub_dataset(args, data_packet):
    # pickle.dump((data_packet['x_train'], data_packet['y_train'], data_packet['x_valid'], data_packet['y_valid'], \
    #     data_packet['x_test'], data_packet['y_test'], data_packet['x_test_assay_list'], data_packet['y_test_assay_list']),\
    #          open(os.path.join(args.data_dir, DTI_file_name), 'wb'))
    # print(f'Construct dataset {DTI_file_name} to {args.data_dir}')
    store_x_y(args, data_packet['x_train'], data_packet['y_train'], 'train')
    store_x_y(args, data_packet['x_valid'], data_packet['y_valid'], 'valid')
    store_x_y(args, data_packet['x_test'], data_packet['y_test'], 'test')
    store_x_y(args, data_packet['x_test_assay_list'], data_packet['y_test_assay_list'], 'test_assay_list')
    

def read_dti_sub_dataset(args):
    data_packet = {}
    # data_packet['x_train'], data_packet['y_train'], data_packet['x_valid'], data_packet['y_valid'], \
    #     data_packet['x_test'], data_packet['y_test'], data_packet['x_test_assay_list'], data_packet['y_test_assay_list']\
    #          = pickle.load(open(os.path.join(args.data_dir, DTI_file_name), 'rb'))
    # print(f'Read dataset {DTI_file_name} from {args.data_dir}')
    data_packet['x_train'], data_packet['y_train'] = read_x_y(args, 'train')
    data_packet['x_valid'], data_packet['y_valid'] = read_x_y(args, 'valid')
    data_packet['x_test'], data_packet['y_test'] = read_x_y(args, 'test')
    data_packet['x_test_assay_list'], data_packet['y_test_assay_list'] = read_x_y(args, 'test_assay_list')
    return data_packet

def get_hparams(args):
    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
        
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))
    return hparams

def update_sub(args,len_list):
    if args.sub_sample_batch_max_num > 0 :
        len_list = [min([min(len_list), args.sub_sample_batch_max_num]) for _ in range(len(len_list))]
    return len_list
    
import sys
def get_Dti_dg_data_packet(args,hparams):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.store_log:
        sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
        sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))
    ENVIRONMENTS = [str(i) for i in list(range(2013, 2022))]
    TRAIN_ENV = [str(i) for i in list(range(2013, 2019))]
    TEST_ENV = ['2019', '2020', '2021']

    idx2train_env = dict(zip(range(len(TRAIN_ENV)), TRAIN_ENV))
    idx2test_env = dict(zip(range(len(TEST_ENV)), TEST_ENV))
    dataset = datasets.TdcDtiDg(args.data_dir, args.test_envs, hparams)

    in_splits = []
    out_splits = []
    uda_splits = []

    test_set = []

    print("constructing in(train)/out(validation) splits with 80%/20% for training dataset")

    for env_i, env in enumerate(dataset):
        #print(f'env_i = {env_i}')
        uda = []
        
        if env_i in args.test_envs:
            ## testing
            out, in_ = misc.split_dataset(env,
                int(len(env)*args.holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))
            
            test_set.append((in_, None))
            
        else:
            ## validation
            out, in_ = misc.split_dataset(env,
                int(len(env)*args.holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

            in_weights, out_weights, uda_weights = None, None, None

            in_splits.append((in_, in_weights))
            out_splits.append((out, out_weights))
        
    print(f'len(in_splits) = {len(in_splits)}, len(out_splits) = {len(out_splits)}')
    print("creating training data loaders...")
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    len_train_loaders = [
        len(env)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs
    ]
    print(f'len_train_loaders = {len_train_loaders}, sum(len_train_loaders) = {sum(len_train_loaders)}')

    print("creating validation data loaders...")
    val_loaders = [FastDataLoader(
        dataset=env,
        #batch_size=256,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for env, _ in (out_splits)]

    len_val_loaders = [
        len(env)
        for env, _ in (out_splits)
    ]
    print(f'len_val_loaders = {len_val_loaders}, sum(len_val_loaders) = {sum(len_val_loaders)}')
    print("creating test data loaders...")

    eval_loaders = [FastDataLoader(
        dataset=env,
        #batch_size=256,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for env, _ in (test_set)]

    len_eval_loaders = [
        len(env)
        for env, _ in (test_set)
    ]

    print(f'len_eval_loaders = {len_eval_loaders}, sum(len_eval_loaders) = {sum(len_eval_loaders)}')

    test_set = test_set[0:2] # 2021 is too less

    in_len= [len(env)/hparams['batch_size'] for env,_ in in_splits]
    out_len = [len(env)/hparams['batch_size'] for env,_ in out_splits]
    test_len = [len(env)/hparams['batch_size'] for env,_ in test_set]

    in_len = update_sub(args,in_len)
    out_len = update_sub(args,out_len)
    test_len = update_sub(args,test_len)

    print(f'sub_sample_batch_max_num = {args.sub_sample_batch_max_num}')
    print(f'in_len = {in_len}, out_len = {out_len},  train_val_sum = {sum(in_len)+sum(out_len)}')
    print(f'test_len = {test_len}, test_sum = {sum(test_len)}')

    return get_data_packet(args, train_loaders,in_len,val_loaders,out_len,eval_loaders,test_len)
    