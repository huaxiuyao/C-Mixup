import string
import numpy as np
import torch



def change_type(x):
    if x == '?':
        return None
    elif x[0].isalpha():
        pass
    else:
        return float(x)

def get_CommunitiesAndCrime_data_packet(args,path = './data/CommunitiesAndCrime/'):

    ########### input ###########
    fboj = open(path + 'communities.data')

    data = []
    flag = 1 # 1 for debug
    for eachline in fboj:
        t = eachline.strip().split(',')
        t.pop(3) # delete county name
        add = [*map(lambda x: -1 if x == '?' else float(x), t)]
        if flag == 1: #debug
            print(t)
            print(add)
            flag = 0
        data.append(add)
    data_tmp = np.array(data)
    
    data_mean = [data_tmp[:,i][ data_tmp[:,i]>= -0.5].mean() for i in range(data_tmp.shape[1])]
    data = np.array([[data_tmp[i,j] if data_tmp[i,j] >= -0.5 else data_mean[j] for j in range(data_tmp.shape[1])] for i in range(data_tmp.shape[0])]) # process nan
    idx2assay = np.array([*map(int,(data[:,0]))])
    
    ####### get shuffle split idx #######
    if args.is_ood == 0:
        shuffle_idx = np.random.permutation(data.shape[0])
        ########## split ###########
        train_num = int(data.shape[0] * args.id_train_val_split[0])
        val_num = int(data.shape[0] * args.id_train_val_split[1])
        test_num = data.shape[0] - train_num - val_num

        train_idx = shuffle_idx[:train_num]
        valid_idx = shuffle_idx[train_num:train_num+val_num]
        test_idx = shuffle_idx[train_num+val_num:]
    else:
        #print(f'idx2assay = {idx2assay}') # [ 8 53 24 ...  9 25  6]
        assay_unique_lookup = np.unique(idx2assay)
        print(f'len(assay_unique_lookup) = {len(assay_unique_lookup)}')
        #print(f'assay_unique_lookup = {assay_unique_lookup}') # [ 1  2  4  5  6  8  9 10 11 12 13 16 18 19 20 21 22 23 24 25 27 28 29 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56]
        assay2idx_list = [torch.nonzero(torch.tensor(idx2assay == loc)).squeeze(-1)
                                    for loc in assay_unique_lookup] # ok
        
        #print(f'assay2idx_list[0] = {assay2idx_list[0]}') # assay2idx_list[0] = tensor([  30,   32,   42,   58,  225,  266,  276,  301,  302,  360,  362,  420, ... 
        assaynum = [len(l)  for l in assay2idx_list] 
        assaynum_sum_forward = [sum(assaynum[:i+1]) for i in range(len(assaynum))]
        
        #print(f'assaynum = {assaynum}') # assaynum = [43, 3, 20, 25, 278, 25, 69, 1, 1, 90, 37, 7, 48, 20, 1, 26, 22, 17, 12, 121, 7, 19, 42, 5, 21, 211, 10, 46, 46, 8, 109, 36, 31, 101, 26, 28, 9, 35, 156, 24, 4, 33, 40, 14, 60, 7]
        print(f'assaynum_sum_forward = {assaynum_sum_forward}')
    # assay
        #print(data[0])

        train_assay_num = int(sum(assaynum) * args.id_train_val_split[0])
        valid_assay_num = int(sum(assaynum) * args.id_train_val_split[1])
        test_assay_num = sum(assaynum) - train_assay_num - valid_assay_num
        idx1, idx2 = -1,-1
        for i in range(len(assaynum_sum_forward)): # disjoint
            if assaynum_sum_forward[i] >= train_assay_num and idx1 == -1:
                idx1 = i
            if idx1 != -1 and assaynum_sum_forward[i] >= assaynum_sum_forward[idx1] + valid_assay_num and idx2 == -1:
                idx2 = i
        print(f'idx1 = {idx1}, idx2 = {idx2}')
        train_idx = [k for sub in assay2idx_list[:idx1] for k in sub]
        valid_idx = [k for sub in assay2idx_list[idx1:idx2] for k in sub]
        test_idx =  [k for sub in assay2idx_list[idx2:] for k in sub]
        
        test_assay2idx_list = [sub for sub in assay2idx_list[idx2:]]

        print(f'len(test_assay2idx_list) = {len(test_assay2idx_list)}')
        # shuffle
        train_idx = np.array(train_idx)[np.random.permutation(len(train_idx))]
        valid_idx = np.array(valid_idx)[np.random.permutation(len(valid_idx))]
        test_idx = np.array(test_idx)[np.random.permutation(len(test_idx))]
        test_assay2idx_list = [np.array(sub)[np.random.permutation(len(sub))] for sub in test_assay2idx_list]

        print(f'len(train_idx) = {len(train_idx)}, len(valid_idx) = {len(valid_idx)}, len(test_idx) = {len(test_idx)}')
        print(f'len(test_assay2idx_list[0,1,2]) = {len(test_assay2idx_list[0]),len(test_assay2idx_list[1]),len(test_assay2idx_list[2])}')

    ########## read x, y, assay and x normalization ########

    x_data = data[:,4:-1] # delete state and county name, just 3 non-predictive
    y_data = data[:,-1]
    y_data = y_data[:,np.newaxis] 

    x_max = np.amax(x_data, axis = 0)
    x_min = np.amin(x_data, axis = 0)
    #print(f'x_max = {x_max}')
    #print(f'x_min = {x_min}')
    x_data = (x_data - x_min) / (x_max - x_min)
    # just scale x, not y
    # communities and crime need not scale

    ########### shuffle and split ############
    x_train = x_data[train_idx]
    x_valid = x_data[valid_idx]
    x_test = x_data[test_idx]

    y_train = y_data[train_idx]
    y_valid = y_data[valid_idx]
    y_test = y_data[test_idx]

    idx2assay_train = idx2assay[train_idx]
    assay2idx_train = {loc:torch.nonzero(torch.tensor(idx2assay_train == loc)).squeeze(-1)
                            for loc in np.unique(idx2assay_train)}
    statisc_assay2idx_train = [len(assay2idx_train[idx]) for idx in assay2idx_train.keys()]
    print(f'statisc_assay2idx_train = {statisc_assay2idx_train}')
    print(f'min(statisc_assay2idx_train) = {min(statisc_assay2idx_train)}, max(statisc_assay2idx_train) = {max(statisc_assay2idx_train)},sum(statisc_assay2idx_train) = {sum(statisc_assay2idx_train)}')
    #print(y_train)
    if args.show_setting:
        print(f'{x_train.shape,x_valid.shape,x_test.shape,y_train.shape,y_valid.shape,y_test.shape}')
        print(f'communities and crime data.shape = {data.shape}')
        print(f'len(idx2assay) = {len(idx2assay)}, idx2assay = {idx2assay}, len(idx2assay_train) = {len(idx2assay_train)}, idx2assay_train = {idx2assay_train}')

    

    idx2assay_test = idx2assay[test_idx]
    data_packet = {
        'x_train': x_train,
        'x_valid': x_valid,
        'x_test': x_test,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test,
    }

    if args.is_ood == True:
        data_packet['idx2assay_train'] = idx2assay_train
        data_packet['assay2idx_train'] = assay2idx_train
        x_test_assay_list = [x_data[tmpidx] for tmpidx in test_assay2idx_list]
        y_test_assay_list = [y_data[tmpidx] for tmpidx in test_assay2idx_list]
        print(f'len(y_test_assay_list) = {len(y_test_assay_list)}')
        data_packet['x_test_assay_list'] = x_test_assay_list
        data_packet['y_test_assay_list'] = y_test_assay_list

        data_packet['idx2assay_test'] = idx2assay_test
        
        if args.show_process:
            print('ood done!')
    else:
        if args.show_process:
            print('id done!')

    print(f'len(assay2idx_train) = {len(assay2idx_train)}')
    print(f"len(x_test_assay_list) = {len(data_packet['x_test_assay_list'])}")
    
    return data_packet

class Args():
    def __init__(self, show_setting, is_ood, show_process):
        self.show_setting = show_setting
        self.is_ood = is_ood
        self.show_process = show_process
        self.id_train_val_split = [0.7,0.1,0.2]

if __name__ == '__main__':
    args = Args(1,1,1)
    data_packet = get_CommunitiesAndCrime_data_packet(args)