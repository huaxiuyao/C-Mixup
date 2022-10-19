import argparse
import datetime
import random
import json
import os
import sys
import csv
from collections import defaultdict
from tempfile import mkdtemp

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import models
from config import dataset_defaults
from utils import unpack_data, save_best_model, \
    Logger, return_predict_fn, return_criterion, save_pred

from mixup import mix_up


# code base: https://github.com/huaxiuyao/LISA/tree/main/domain_shifts

runId = datetime.datetime.now().isoformat().replace(':', '_')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Gradient Matching for Domain Generalization.')
# General
parser.add_argument('--dataset', type=str, default='poverty',
                    help="Name of dataset")
parser.add_argument('--algorithm', type=str, default='erm',
                    help='training scheme, choose between fish or erm.')
parser.add_argument('--experiment', type=str, default='.',
                    help='experiment name, set as . for automatic naming.')
parser.add_argument('--experiment_dir', type=str, default='../',
                    help='experiment directory')
parser.add_argument('--data-dir', type=str, default='./',
                    help='path to data dir')
# Computation
parser.add_argument('--nocuda', type=int, default=False,
                    help='disables CUDA use')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed, set as -1 for random.')
parser.add_argument("--mix_alpha", default=2, type=float)
parser.add_argument("--print_loss_iters", default=100, type=int)
parser.add_argument("--kde_bandwidth", default=0.5, type=float)

parser.add_argument("--is_kde", default=0, type=int) # kde mixup or random mixup
parser.add_argument("--save_pred", default=False, action='store_true')
parser.add_argument("--save_dir", default='result', type=str)
parser.add_argument("--fold", default='A', type=str)

args = parser.parse_args()
args.cuda = not args.nocuda and torch.cuda.is_available()
device = torch.device("cuda")
if args.nocuda:
    print(f'use cpu')
    device = torch.device("cpu")

args_dict = args.__dict__
args_dict.update(dataset_defaults[args.dataset]) # default configuration
args = argparse.Namespace(**args_dict)

# random select a training fold according to seed. Can comment this line and set args.fold manually as well
args.fold = ['A', 'B', 'C', 'D', 'E'][args.seed % 5] 

if args.save_pred:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

##### set seed #####
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
print(f'args.seed = {args.seed}')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# experiment directory setup
args.experiment = f"{args.dataset}_{args.algorithm}_{args.seed}" \
    if args.experiment == '.' else args.experiment
if args.is_kde:
    args.experiment += f'_kde_bw{args.kde_bandwidth}'
directory_name = '{}/experiments/{}'.format(args.experiment_dir,args.experiment)

if not os.path.exists(directory_name):
    os.makedirs(directory_name)
runPath = mkdtemp(prefix=runId, dir=directory_name)

# logging setup
sys.stdout = Logger('{}/run.log'.format(runPath))
print('RunID:' + runPath)
print(args)
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
torch.save(args, '{}/args.rar'.format(runPath))

# load model
modelC = getattr(models, args.dataset)
if args.algorithm == 'mixup': args.batch_size //= 2

train_loader, tv_loaders = modelC.getDataLoaders(args, device=device)
val_loader, test_loader = tv_loaders['val'], tv_loaders['test']
model = modelC(args, weights=None).to(device)

print(f'len(train_loader) = {len(train_loader)}, len(val_loader) = {len(val_loader)}, len(test_loader) = {len(test_loader)}')

n_class = getattr(models, f"{args.dataset}_n_class")

assert args.optimiser in ['SGD', 'Adam', 'AdamW'], "Invalid choice of optimiser, choose between 'Adam' and 'SGD'"
opt = getattr(optim, args.optimiser)

params = filter(lambda p: p.requires_grad, model.parameters())
optimiserC = opt(params, **args.optimiser_args)

predict_fn, criterion = return_predict_fn(args.dataset), return_criterion(args.dataset)

def train_erm(train_loader, epoch, agg):
    running_loss = 0
    total_iters = len(train_loader)
    print('\n====> Epoch: {:03d} ,arg = erm'.format(epoch))
    for i, data in enumerate(train_loader):
        model.train()
        # get the inputs
        x, y = unpack_data(data, device)
        optimiserC.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        if args.use_bert_params:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.max_grad_norm)
        optimiserC.step()

        running_loss += loss.item()
        # print statistics
        if (i + 1) % args.print_loss_iters == 0 and args.print_iters != -1 :
            agg['train_loss'].append(running_loss / args.print_loss_iters)
            agg['train_iters'].append(i + 1 + epoch * total_iters)
            print(
                'iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_loss_iters))
            running_loss = 0.0

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg, args)


def train_mixup(train_loader, epoch, agg):
    print('into train_mixup')
    model.train()
    train_loader.dataset.reset_batch()
    print('\n====> Epoch: {:03d} '.format(epoch))

    # The probabilities for each group do not equal to each other.
    for i, data in enumerate(train_loader):
        model.train()
        x1, y1, g1, prev_idx = data[0].to(device), data[1].to(device), data[2].to(device), data[3]
        if y1.ndim > 1:
            y1 = y1.squeeze()
        x2, y2, g2 = [], [], []
        
        assert train_loader.dataset.num_envs > 1
        for g, y, idx in zip(g1,y1, prev_idx):
            tmp_x, tmp_y, tmp_g = train_loader.dataset.get_sample(idx.item(), UseKDE = args.is_kde, y1=y)
            x2.append(tmp_x.unsqueeze(0))
            y2.append(tmp_y)
            g2.append(tmp_g)

        x2 = torch.cat(x2).to(device)
        y2 = torch.cat(y2).to(device)

        loss_fn = torch.nn.MSELoss()
        # mixup
        mixed_x1, mixed_y1 = mix_up(args, x1, y1, x2, y2, args.dataset)
        mixed_x2, mixed_y2 = mix_up(args, x2, y2, x1, y1, args.dataset)

        mixed_x = torch.cat([mixed_x1, mixed_x2])
        mixed_y = torch.cat([mixed_y1, mixed_y2])

        # forward
        outputs = model(mixed_x)

        loss = loss_fn(outputs, mixed_y)

        # backward
        optimiserC.zero_grad()
        loss.backward()
        optimiserC.step()

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            print(f'iteration {(i + 1):05d}: ')
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg, args)
            model.train()

def test(test_loader, agg, loader_type='test', verbose=True, save_ypred=False, save_dir=None):
    model.eval()
    yhats, ys, metas = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # get the inputs
            x, y = batch[0].to(device), batch[1].to(device)
            y_hat = model(x)
            ys.append(y)
            yhats.append(y_hat)
            metas.append(batch[2])

        ypreds, ys, metas = predict_fn(torch.cat(yhats)), torch.cat(ys), torch.cat(metas)
        if save_ypred: # random select a fold
            save_name = f"{args.dataset}_split:{loader_type}_fold:" \
                        f"{['A', 'B', 'C', 'D', 'E'][args.seed % 5]}" \
                        f"_epoch:best_pred.csv"
            with open(f"{runPath}/{save_name}", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(ypreds.unsqueeze(1).cpu().tolist())
        test_val = test_loader.dataset.eval(ypreds.cpu(), ys.cpu(), metas)

        agg[f'{loader_type}_stat'].append(test_val[0][args.selection_metric])
        if verbose:
            print(f"=============== {loader_type} ===============\n{test_val[-1]}")


if __name__ == '__main__':
    # set learning rate schedule
    if args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimiserC,step_size = 1, **args.scheduler_kwargs)
        scheduler.step_every_batch = False
        scheduler.use_metric = False
    else:
        scheduler = None

    print("=" * 30 + f" Training: {args.algorithm} for {args.dataset} "  + "=" * 30)
    
    train = locals()[f'train_{args.algorithm}'] 
    agg = defaultdict(list)
    agg['val_stat'] = [0.]
    agg['test_stat'] = [0.]

    for epoch in range(args.epochs):
        train(train_loader, epoch, agg)
        test(val_loader, agg,'val',True)
        if scheduler is not None:
            scheduler.step()

        test(test_loader, agg,'test', True)
        save_best_model(model, runPath, agg, args)

        if args.save_pred:
            save_pred(args,model, train_loader, epoch, args.save_dir,predict_fn,device)

    model.load_state_dict(torch.load(runPath + '/model.rar'))
    print('Finished training! Loading best model...')
    for split, loader in tv_loaders.items():
        test(loader, agg, loader_type=split,verbose=True, save_ypred=True)
