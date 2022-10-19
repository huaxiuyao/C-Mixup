from cgi import test
from torchvision.datasets import FashionMNIST
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from torch.utils.data import random_split,DataLoader
import os
import copy

import PIL.Image as Image
import random
from scipy.stats import pearsonr
import pickle


data_transforms = {
    'train': transforms.Compose([
        #transforms.Resize(28),
        # transforms.RandomHorizontalFlip(),
        transforms.Grayscale(3), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        #transforms.Resize(28),
        transforms.Grayscale(3),
        transforms.ToTensor(), 
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

def img_torch2numpy(img): # N C H W --> N H W C
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if len(img.shape) == 4:
        return np.transpose(npimg, (0, 2, 3, 1))
    elif len(img.shape) == 3:
        return np.transpose(npimg, (1, 2, 0))

def img_numpy2torch(img):# N H W C --> N C H W  
    if len(img.shape) == 4:
        tmp_img = np.transpose(img, (0, 3, 1 ,2))
    else:
        tmp_img = np.transpose(img, (2, 0 ,1))
    tcimg = torch.tensor(tmp_img)
    tcimg = (tcimg - 0.5) * 2 # normalize
    return tcimg

def img_save(img, file_name):
    
    plt.switch_backend('agg')
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    img = img / 2 + 0.5     # unnormalize
    if not isinstance(img, np.ndarray):
        npimg = img.numpy()
    else:
        npimg = img
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    plt.savefig(f'./pic2/{file_name}.jpg')

rotate_class = torch.tensor([ (360.0 / 60) * i for i in range(60) ])

def get_random_degree(use_rotate_class=False):
    if use_rotate_class:
        return torch.tensor(random.sample(rotate_class,1 ))
    else:
        return torch.rand(1) * 360.0

# rotate a image by PIL
def rotate_img(args, img, degree = None):
    img = img / 2 + 0.5     # unnormalize
    pil_img = transforms.ToPILImage()(img)
    if degree == None:
        degree = get_random_degree(args.use_rotate_class)
    r_img = pil_img.rotate(degree)
    r_img = transforms.ToTensor()(r_img) # read float
    r_img = (r_img - 0.5) * 2.0     # normalize
    return r_img, degree

# rotate set of images
def get_rotate_imgs(args, imgs:torch.Tensor, name): # return numpy !
    r_img_list, degree_list = [], []
    for i in range(imgs.shape[0]):
        r_img, degree = rotate_img(args, imgs[i])
        degree /= 360.0 # critical --> normalize y
        r_img_list.append(r_img.unsqueeze(0))
        degree_list.append(degree)

    r_img_list = torch.cat(r_img_list, dim=0)
    degree_list = torch.cat(degree_list, dim=0)

    print(f'finish rotate for {name}')
    print(f'r_img_list.shape = {r_img_list.shape}')
    r_img_np = img_torch2numpy(r_img_list)
    degree_np = degree_list.numpy()

    
    return r_img_np, degree_list

def get_all_batches(loader, name, classes):

    data_iter = iter(loader)

    # for debug
    steplen = len(loader)

    img_list = []
    label_list = []

    for step in range(steplen):
        #print(f'step = {step}')
        
        if step != 0 and step % (steplen // 5) == 0:
            print('For {} dataloader: get whole packet : {:.3f}% ({} / {})'.format(name,step*100/(steplen), step, steplen))
        images, labels = data_iter.next()
        
        img_list.append(images)
        label_list.append(labels)

    img_tensor = torch.cat(img_list, dim = 0)
    label_tensor = torch.cat(label_list, dim = 0)
    print(f'type(img_tensor) = {type(img_tensor)}, img_tensor.shape = {img_tensor.shape}, label_tensor.shape = {label_tensor.shape}')
    
    return img_tensor, label_tensor

def test_img(img, x, degree, name, iscolor = True):
    for i in range(3):
        img_save(img[i], f'{name}_{i}')
        if iscolor:
            img_save(x[i], f'{name}_{i}_color_r_{float(degree[i])}')
        else:
            img_save(x[i], f'{name}_{i}_no_color_r_{float(degree[i])}')

def store_color(args, x, y, name, is_color = True):
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    
    exp_str = '_all_pos_color' if args.all_pos_color == 1 else ''
    if is_color:
        pickle.dump((x, y), open(os.path.join(args.data_dir, f"color_{name}_data_ratio_{args.spurious_ratio}{exp_str}.pkl"), 'wb'))
    else:
        pickle.dump((x, y), open(os.path.join(args.data_dir, f"no_color_{name}_data.pkl"), 'wb'))
    print(f'finish store {name}')

def store_ood(args, idx2assay_train, assay2idx_train, x_test_assay_list, y_test_assay_list):
    pickle.dump((idx2assay_train, assay2idx_train, x_test_assay_list, y_test_assay_list), open(os.path.join(args.data_dir, f"ood_data_ratio_{args.spurious_ratio}.pkl"), 'wb'))
    print(f'finish store ood')

def read_color(args, name, is_color = True): # is_color = false -> read r-fmnist; = true -> read rc-fmnist
    exp_str = '_all_pos_color' if args.all_pos_color == 1 else ''
    if is_color:
        x, y = pickle.load(open(os.path.join(args.data_dir, f"color_{name}_data_ratio_{args.spurious_ratio}{exp_str}.pkl"), 'rb'))
    else:
        x, y = pickle.load(open(os.path.join(args.data_dir, f"no_color_{name}_data.pkl"), 'rb'))
    print(f'read {name} !')
    return x,y

def read_ood(args):
    idx2assay_train, assay2idx_train, x_test_assay_list, y_test_assay_list = pickle.load(open(os.path.join(args.data_dir, f"ood_data_ratio_{args.spurious_ratio}.pkl"), 'rb'))
    print(f'read ood')
    return idx2assay_train, assay2idx_train, x_test_assay_list, y_test_assay_list 

def visualization(args, imgs):
    degree_list = [0.0, 90.0, 180.0, 270.0, 359.9]#[0.0,120.0,240.0,359.9]#[0.0, 90.0, 180.0, 270.0, 359.9]
    i_range = [2,7,9]
    for i in i_range:
        img = imgs[i]#[:3]
        for d in degree_list:
            r_img, _ = rotate_img(args, img, d)
            img_save(r_img, f'vis_{i}_{d}_ori')
            print(f'r_img.shape = {r_img.shape}')
            r_img_np = img_torch2numpy(r_img)
            r_img_np2 = copy.deepcopy(r_img_np)
            r_img_np3 = copy.deepcopy(r_img_np)
            d_np = d / 360
            print(f'r_img_np.shape = {r_img_np.shape}, d_np = {d_np}')
            
            rc_img = red_blue_linear_map(r_img_np, d_np)        
            rc_img_inv = red_blue_linear_map(r_img_np2, 1 - d_np)    
            rc_img_rd = red_blue_linear_map(r_img_np3, random.uniform(0,1))

            rc_img_torch = img_numpy2torch(rc_img)  
            rc_img_inv_torch = img_numpy2torch(rc_img_inv)  
            rc_img_rd_torch = img_numpy2torch(rc_img_rd)

            img_save(rc_img_torch, f'vis_{i}_{d}_pos')
            img_save(rc_img_inv_torch, f'vis_{i}_{d}_inv')
            img_save(rc_img_rd_torch, f'vis_{i}_{d}_rd')


def get_RCF_MNIST_data_packet(args):

    if (args.construct_color_data == 1) or (args.construct_no_color_data == 1) or (args.vis_rcf == 1):
        assert ( (args.construct_color_data == 1) and (args.construct_no_color_data == 1)) == False
        train_set = FashionMNIST(root='./data', train=True, 
                                download=True, transform=data_transforms['train'])

        train_set, val_set, test_set = random_split(train_set, (40000, 10000, 10000))

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                                                num_workers=0)

        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, 
                                                num_workers=0)

        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, 
                                                num_workers=0)

        classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


        image_datasets = {'train': train_set, 'val': val_set, 'test': test_set}
        dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

        print(f'dataset_sizes = {dataset_sizes}')

        # concatenate all batches
        
        img_val, label_val = get_all_batches(val_loader,'val', classes)
        

        if args.vis_rcf == 1:
            visualization(args, img_val)
            exit(0)
        
        img_train, label_train = get_all_batches(train_loader,'train', classes)
        img_test, label_test = get_all_batches(test_loader,'test', classes)

        r_img_train, degree_train = get_rotate_imgs(args, img_train, 'train')
        r_img_val, degree_val = get_rotate_imgs(args, img_val, 'val')
        r_img_test, degree_test = get_rotate_imgs(args, img_test, 'test')

        print(f'r_img_val.shape = {r_img_val.shape}, degree_val.shape = {degree_val.shape}')

        if args.construct_color_data:
            assert args.is_ood == 1
            color_r_img_train, color_r_img_val, color_r_img_test, idx2assay_train, assay2idx_train, test_assay2idx_list = linear_red_blue_preparation(args, r_img_train, r_img_val, r_img_test, degree_train, degree_val, degree_test)
        else:
            color_r_img_train, color_r_img_val, color_r_img_test = r_img_train, r_img_val, r_img_test
        
        # recover to torch
        x_train = img_numpy2torch(color_r_img_train)
        x_val = img_numpy2torch(color_r_img_val)
        x_test = img_numpy2torch(color_r_img_test)

        test_assay2idx_list = [np.array(sub)[np.random.permutation(len(sub))] for sub in test_assay2idx_list]
        x_test_assay_list = [x_test[tmpidx] for tmpidx in test_assay2idx_list]
        y_test_assay_list = [degree_test[tmpidx] for tmpidx in test_assay2idx_list]

        # test img by visualizing
        test_img(img_train, x_train, degree_train, 'train', args.construct_color_data)
        test_img(img_val, x_val, degree_val, 'val', args.construct_color_data)
        test_img(img_test, x_test, degree_test, 'test', args.construct_color_data)

        # store into ./data
        store_color(args, x_train, degree_train, 'train', args.construct_color_data)
        store_color(args, x_val, degree_val, 'val', args.construct_color_data)
        store_color(args, x_test, degree_test, 'test', args.construct_color_data)
        store_ood(args, idx2assay_train, assay2idx_train, x_test_assay_list, y_test_assay_list)
    else:
        assert ( (args.construct_color_data == -1) and (args.construct_no_color_data == -1)) == False
        assert ( (args.construct_color_data == -1) or (args.construct_no_color_data == -1)) == True

        x_train, degree_train = read_color(args, 'train', (args.construct_color_data == -1))
        x_val, degree_val = read_color(args, 'val', (args.construct_color_data == -1))
        x_test, degree_test = read_color(args, 'test', (args.construct_color_data == -1))
        if args.is_ood:
            idx2assay_train, assay2idx_train, x_test_assay_list, y_test_assay_list  = read_ood(args)

    data_packet = {
        'x_train': x_train,
        'x_valid': x_val,
        'x_test': x_test,
        'y_train': degree_train.unsqueeze(1), # for dimension match
        'y_valid': degree_val.unsqueeze(1),
        'y_test': degree_test.unsqueeze(1),
        # 'idx2assay_train': idx2assay_train,
        # 'assay2idx_train': assay2idx_train,
        # 'x_test_assay_list': x_test_assay_list,
        # 'y_test_assay_list': y_test_assay_list
    }
    if args.is_ood:
        data_packet['idx2assay_train'] = idx2assay_train
        data_packet['assay2idx_train'] = assay2idx_train
        
        print(f'len(y_test_assay_list) = {len(y_test_assay_list)}')
        data_packet['x_test_assay_list'] = x_test_assay_list
        data_packet['y_test_assay_list'] = y_test_assay_list

    return data_packet


def copydim( set:np.array, num=3):
    return np.expand_dims(set,-1).repeat(num,axis=-1)

# def linear_red_blue_preparation(args, x_val, y_val ):
def linear_red_blue_preparation(args, x_train, x_val, x_test, y_train, y_val, y_test):
    # # calculate the pearson between ratio and y

    x_train, ratio_reshape, y_reshape, idx2assay_train, assay2idx_train, _ = color_linear_red_blue(x_train, y_train, args.spurious_ratio, use_spurious = True, inv = False)
    corr = pearsonr(ratio_reshape, y_reshape)
    print(f'corr of train = {corr}')
    
    x_val, ratio_reshape, y_reshape,_, _, _= color_linear_red_blue(x_val, y_val, args.spurious_ratio, use_spurious = False)
    corr = pearsonr(ratio_reshape, y_reshape)
    print(f'corr of val = {corr}')

    test_inv = (args.all_pos_color != 1) # not all positive
    x_test, ratio_reshape, y_reshape,_, _, test_assay2idx_list  = color_linear_red_blue(x_test, y_test, args.spurious_ratio, use_spurious = True, inv = test_inv)
    corr = pearsonr(ratio_reshape, y_reshape)
    print(f'corr of test = {corr}')

    return x_train, x_val, x_test, idx2assay_train, assay2idx_train, test_assay2idx_list#, x_test_assay_list, y_test_assay_list 


def color_linear_red_blue(x_set:np.array, y_set:np.array, spurious_ratio = 0.9, use_spurious = 1, inv = False):
    y_reshape = y_set # cannot reshape(-1), it will inv!
    x_tmp = x_set
    ratio_reshape = np.zeros_like(y_reshape)

    print(f'x_tmp.shape = {x_tmp.shape}, y_reshape.shape = {y_reshape.shape}')

    num = int(y_reshape.shape[0])
    idx = np.arange(num)
    idx2assay = np.zeros(num)

    if use_spurious: # spurious
        mixtype_normal_idx = np.random.choice(idx, size=int(num * spurious_ratio), replace=False)

        mixtype_inverse_idx = np.setdiff1d(idx, mixtype_normal_idx)
        ratio_matric = copydim(copydim(y_reshape,1),1)

        if inv == False: # normal spurious correlation
            
            x_tmp[mixtype_normal_idx] = red_blue_linear_map(x_tmp[mixtype_normal_idx], ratio_matric[mixtype_normal_idx])
            x_tmp[mixtype_inverse_idx] = red_blue_linear_map(x_tmp[mixtype_inverse_idx], 1.0 - ratio_matric[mixtype_inverse_idx])
            ratio_reshape[mixtype_normal_idx] = y_reshape[mixtype_normal_idx]
            ratio_reshape[mixtype_inverse_idx] = 1.0 - y_reshape[mixtype_inverse_idx]

        else: # inverse spurious correlation
            x_tmp[mixtype_normal_idx] = red_blue_linear_map(x_tmp[mixtype_normal_idx], 1.0 - ratio_matric[mixtype_normal_idx])
            x_tmp[mixtype_inverse_idx] = red_blue_linear_map(x_tmp[mixtype_inverse_idx], ratio_matric[mixtype_inverse_idx])
            ratio_reshape[mixtype_normal_idx] = 1.0 - y_reshape[mixtype_normal_idx]
            ratio_reshape[mixtype_inverse_idx] = y_reshape[mixtype_inverse_idx]
        
        corr = pearsonr(ratio_reshape, y_reshape)
        print(f'inv = {inv}, corr = {corr}, ratio_reshape = {ratio_reshape}, y_reshape = {y_reshape}')

        
        idx2assay[mixtype_normal_idx] = 0 # class 0
        idx2assay[mixtype_inverse_idx] = 1 # class 1

    else: # test random
        ratio_reshape = np.random.rand(num)
        ratio = copydim(copydim(ratio_reshape,1),1)
        x_tmp = red_blue_linear_map(x_tmp, ratio)
        # all class 0

    x_set = x_tmp

    print(f'ratio_reshape = {ratio_reshape}')

    # ood
    assay2idx_list = [torch.nonzero(torch.tensor(idx2assay == loc)).squeeze(-1)
                                for loc in np.unique(idx2assay)] # ok
    assay2idx = {loc:torch.nonzero(torch.tensor(idx2assay == loc)).squeeze(-1)
                            for loc in np.unique(idx2assay)}
    
    return x_set, ratio_reshape, y_reshape, idx2assay, assay2idx, assay2idx_list

color_up_bound = 100 / 255
color_lower_bound = 5 / 255 # 60 / 255

def red_blue_linear_map(imgt:np.array, red_ratio):
    print(f'imgt[14,:,0] = {imgt[14,:,0]}, red_ratio = {red_ratio}')
    # R outside background -> red * ratio
    imgt[...,0] = np.where(imgt[...,0] > color_lower_bound , imgt[...,0] * red_ratio, imgt[...,0])
    # imgt[...,0] = np.where(imgt[...,0] < color_up_bound , imgt[...,0] * red_ratio, imgt[...,0])
    # G outside background -> 0
    imgt[...,1] = np.where(imgt[...,1] > color_lower_bound , 0, imgt[...,1])
    # imgt[...,1] = np.where(imgt[...,1] < color_up_bound , 0, imgt[...,1])
    # B outside background -> blue * (1 - ratio)
    imgt[...,2] = np.where(imgt[...,2] > color_lower_bound , imgt[...,2] * (1 - red_ratio), imgt[...,2])
    # imgt[...,2] = np.where(imgt[...,2] < color_up_bound , imgt[...,2] * (1 - red_ratio), imgt[...,2])
    return imgt