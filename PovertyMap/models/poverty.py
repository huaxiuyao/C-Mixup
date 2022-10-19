import os
from copy import deepcopy
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from wilds.common.data_loaders import get_eval_loader
from wilds.datasets.poverty_dataset import PovertyMapDataset

from .resnet_multispectral import ResNet18

import torch
from torch.utils.data import Dataset
from sklearn.neighbors import KernelDensity

# code base: https://github.com/p-lambda/wilds

class Poverty_Batched_Dataset(Dataset):
    """
    Batched dataset for Poverty. Allows for getting a batch of data given
    a specific domain index.
    """
    def __init__(self, dataset, split, batch_size, transform=None, args = None):
        self.split_array = dataset.split_array
        self.split_dict = dataset.split_dict
        self.args = args
        split_mask = self.split_array == self.split_dict[split]
        # split_idx = [ 2390  2391  2392 ... 19665 19666 19667]
        self.split_idx = np.where(split_mask)[0] 

        self.root = dataset.root
        self.no_nl = dataset.no_nl

        self.metadata_array = torch.stack([dataset.metadata_array[self.split_idx, i] for i in [0, 2]], -1)
        
        self.y_array = dataset.y_array[self.split_idx]

        self.eval = dataset.eval
        self.collate = dataset.collate
        # metadata_fields:['urban', 'y', 'country']
        self.metadata_fields = dataset.metadata_fields
        self.data_dir = dataset.data_dir

        self.transform = transform if transform is not None else lambda x: x

        domains = self.metadata_array[:, 1] # domain column for csv

        self.domain_indices = [torch.nonzero(domains == loc).squeeze(-1)
                               for loc in domains.unique()] # split into different domain idx, domain_indices is 2D
        # visualization #
        print('='*20 + f' Data information ' + '='*20)
        print(f'len(self.domain_indices):{len(self.domain_indices)}')

        self.domains = domains
        self.domain2idx = {loc.item(): i for i, loc in enumerate(self.domains.unique())}
        
        self.num_envs = len(domains.unique())
        self.targets = self.y_array
        self.batch_size = batch_size

    def get_sample(self, prev_idx, UseKDE = False, y1 = None):
        d_idx = [d_i for l in self.domain_indices for d_i in l] # all the list

        if UseKDE:
            domain_target_list = self.targets[d_idx]

            kde_patch = KernelDensity(kernel = 'gaussian', bandwidth=self.args.kde_bandwidth).fit([[y1.cpu()]])#should be 2D
            domain_each_rate = np.exp(kde_patch.score_samples(domain_target_list))
            domain_each_rate /= domain_each_rate.sum() #norm

            idx = np.random.choice(d_idx,p = domain_each_rate)
            while idx == prev_idx and len(d_idx) > 1:
                idx = np.random.choice(d_idx,p = domain_each_rate)
        else: # random select index
            idx = np.random.choice(d_idx)
            while idx == prev_idx and len(d_idx) > 1:
                idx = np.random.choice(d_idx)

        return self.transform(self.get_input(idx)), self.targets[idx], self.domains[idx]


    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, d_idx in enumerate(self.domain_indices):
            self.batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_input(self, idx):
        """Returns x for a given idx."""
        img = np.load(self.root / 'images' / f'landsat_poverty_img_{self.split_idx[idx]}.npz')['x']
        if self.no_nl:
            img[-1] = 0
        img = torch.from_numpy(img).float()
        return img


    def __getitem__(self, idx):
        return self.transform(self.get_input(idx)), \
               self.targets[idx], self.domains[idx], idx

    def __len__(self):
        return len(self.targets)



IMG_HEIGHT = 224
NUM_CLASSES = 1
target_resolution = (224, 224)


def initialize_poverty_train_transform():
    """Adapted from the Wilds library, available at: https://github.com/p-lambda/wilds"""

    def ms_cutout(ms_img):
        def _sample_uniform(a, b):
            return torch.empty(1).uniform_(a, b).item()

        assert ms_img.shape[1] == ms_img.shape[2]
        img_width = ms_img.shape[1]
        cutout_width = _sample_uniform(0, img_width/2)
        cutout_center_x = _sample_uniform(0, img_width)
        cutout_center_y = _sample_uniform(0, img_width)
        x0 = int(max(0, cutout_center_x - cutout_width/2))
        y0 = int(max(0, cutout_center_y - cutout_width/2))
        x1 = int(min(img_width, cutout_center_x + cutout_width/2))
        y1 = int(min(img_width, cutout_center_y + cutout_width/2))

        # Fill with 0 because the data is already normalized to mean zero
        ms_img[:, x0:x1, y0:y1] = 0
        return ms_img

    #transform_step = get_image_base_transform_steps()
    transforms_ls = [

        transforms.ToPILImage(),
        transforms.Resize(target_resolution),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(size=target_resolution,),
        transforms.RandomVerticalFlip(),
        #wyp add affine,ms_color and ms_img
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=0.1, scale=(0.9, 1.1)),
        #transforms.Lambda(lambda ms_img: poverty_color_jitter(ms_img)),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1),
        transforms.Lambda(lambda ms_img:ms_cutout(ms_img)),

        transforms.ToTensor()]
    rgb_transform = transforms.Compose(transforms_ls)
    
    return transforms.Compose([]) # empty transform


class Model(nn.Module):
    def __init__(self, args, weights):
        super(Model, self).__init__()
        self.num_classes = NUM_CLASSES

        #resnet18_ms
        self.enc = ResNet18(num_classes=1, num_channels=8)
        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))


    @staticmethod
    def getDataLoaders(args, device):
        kwargs = {'no_nl': False,
                  'fold': args.fold,
                  #'oracle_training_set': False,
                  'use_ood_val': True}
        dataset = PovertyMapDataset(root_dir=os.path.join(args.data_dir, 'wilds'),
                                    download=True, **kwargs)
        # get all train data
        transform = initialize_poverty_train_transform()

        train_sets = Poverty_Batched_Dataset(dataset, 'train', args.batch_size, transform, args)
        datasets = {}
        for split in dataset.split_dict:
            if split != 'train':
                datasets[split] = dataset.get_subset(split, transform=transform)

        kwargs = {'num_workers': 4, 'pin_memory': True} \
            if device.type == "cuda" else {}
        train_loaders = DataLoader(train_sets, shuffle=True, # Shuffle training dataset
                                sampler=None, collate_fn=train_sets.collate,batch_size=args.batch_size, **kwargs)

        print(f'args.fold = {args.fold}')
        print(f'len(train_sets) = {len(train_sets)}')
        tv_loaders = {}
        for split, dataset in datasets.items():
            print(f'len(datasets[{split}]) = {len(dataset)}')
            tv_loaders[split] = get_eval_loader('standard', dataset, batch_size=64) #256)
        return train_loaders, tv_loaders

    def forward(self, x):
        return self.enc(x)
