"""Functions for training and running EF prediction."""

import math
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm

import echonet
from sklearn.neighbors import KernelDensity

@click.command("video")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--task", type=str, default="EF")
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.video.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.video.__dict__[name]))),
    default="r2plus1d_18")
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test", default=False)
@click.option("--num_epochs", type=int, default=45)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--lr_step_period", type=int, default=15)
@click.option("--frames", type=int, default=32)
@click.option("--period", type=int, default=2)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=2)#4)
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)

@click.option("--algorithm", type=str, default='ERM')
@click.option("--kde_type",type=str,default='gaussian')
@click.option("--mix_alpha", type=float, default=0.2)
@click.option("--bandwidth", type=float, default=1e-1)
@click.option("--seed", type=int, default=0)
@click.option("--mixtype", type=str, default='erm')



def run(
    data_dir=None,
    output=None,
    task="EF",

    model_name="r2plus1d_18",
    pretrained=True,
    weights=None,

    run_test=False,
    num_epochs=45,
    lr=1e-4,
    weight_decay=1e-4,
    lr_step_period=15,
    frames=32,
    period=2,
    num_train_patients=None,
    num_workers=0, #2,
    batch_size=10, #20,
    device='cuda', #None,
    seed=0,

    mixtype='erm',
    algorithm='ERM',
    kde_type = 'gaussian',
    bandwidth = 1e-1,
    mix_alpha = 0.2,
    

):
    data_dir = "../../EchoNet-Dynamic/"
    """Trains/tests EF prediction model.
    
    \b
    Args:
        data_dir (str, optional): Directory containing dataset. Defaults to
            `echonet.config.DATA_DIR`.
        output (str, optional): Directory to place outputs. Defaults to
            output/video/<model_name>_<pretrained/random>/.
        task (str, optional): Name of task to predict. Options are the headers
            of FileList.csv. Defaults to ``EF''.
        model_name (str, optional): Name of model. One of ``mc3_18'',
            ``r2plus1d_18'', or ``r3d_18''
            (options are torchvision.models.video.<model_name>)
            Defaults to ``r2plus1d_18''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to True.
        weights (str, optional): Path to checkpoint containing weights to
            initialize model. Defaults to None.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
        num_epochs (int, optional): Number of epochs during training.
            Defaults to 45.
        lr (float, optional): Learning rate for SGD
            Defaults to 1e-4.
        weight_decay (float, optional): Weight decay for SGD
            Defaults to 1e-4.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            Defaults to 15.
        frames (int, optional): Number of frames to use in clip
            Defaults to 32.
        period (int, optional): Sampling period for frames
            Defaults to 2.
        n_train_patients (int or None, optional): Number of training patients
            for ablations. Defaults to all patients.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        device (str or None, optional): Name of device to run on. Options from
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            Defaults to ``cuda'' if available, and ``cpu'' otherwise.
        batch_size (int, optional): Number of samples to load per batch
            Defaults to 20.
        seed (int, optional): Seed for random number generator. Defaults to 0.
    """

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("../../output", "video", "Seed{}_{}_{}_{}_{}_{}_{}_{}".format(seed, mixtype, ('_BW_' + str(bandwidth)) if mixtype == 'kde' else ''
                            ,algorithm, model_name, frames, period, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)
    
    # Set device for computations
    #os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == 'cpu':
        device = torch.device("cpu")
    elif device == 'cuda':
        device = torch.device("cuda")
    
    print(locals())
    print(f'device = {device}, batch_size = {batch_size}, num_epochs = {num_epochs}, num_workers = {num_workers}')
    print(f'mixtype = {mixtype}, algorithm = {algorithm}, kde_type = {kde_type}, bandwidth = {bandwidth}')
    print(f'output = {output}')
    print(f'run_test = {run_test}')
    # Set up model

    model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)

    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    print('finish model definition')

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
    kwargs = {"target_type": task,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }

    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs, pad=12)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)

    dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)
    dataset["test"] = echonet.datasets.Echo(root=data_dir, split="test", **kwargs)

    len_list_dic = {}
    for phase in ["train","val","test"]:
        len_list_dic[phase] = len(dataset[phase])
        len_list_dic[phase] -= (len_list_dic[phase] % batch_size) # extra part

    # len_list_dic['train'] = 100 # debug

    y_list_dic = {}

    ### get y dataset [for length] ###
    flag = 0
    dataloader = torch.utils.data.DataLoader(
                            dataset["train"], batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
    data = []
    with tqdm.tqdm(total=len(dataloader)) as pbar:
        pbar.set_description(f'Get train datapacket, len = {len(dataloader)}')
        for (X, outcome) in dataloader:
            # # debug
            # if flag == 1:
            #     print(f'outcome = {outcome}, outcome.shape = {outcome.shape}')
            #     print(f'X.shape = {X.shape}, all data num in train = {len(dataloader) * X.shape[0]}')
            # # debug
            # if flag >= 10:  break

            data.append(outcome)
            
            pbar.set_postfix_str('outcome[0] = {:.5f}'.format(outcome[0]))
            pbar.update()

            flag += 1

    y_list_dic["train"] = torch.cat(data)
    
    is_mixup = mixtype != 'erm'
    if is_mixup:
        print(f'calculate mixup rate')
        mixup_idx_sample_rate = get_mixup_rate(y_list_dic["train"],mixtype,kde_type,bandwidth,device)
    else:
        mixup_idx_sample_rate = None
        
    
    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        f.write(f'bw = {bandwidth}\n')
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]

                loss, yhat, y = echonet.utils.video.my_run_epoch(model, mixtype, len_list_dic[phase], ds, mix_alpha,batch_size, phase == 'train', optim, device,  mixup_idx_sample_rate)
                f.write("{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                              phase,
                                                              loss,
                                                              sklearn.metrics.r2_score(y, yhat),
                                                              time.time() - start_time,
                                                              y.size,
                                                              sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                              sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                              batch_size,
                                                              bandwidth))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': loss,
                'r2': sklearn.metrics.r2_score(y, yhat),
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            print(f'save model to {output + "checkpoint.pt"}')
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
            f.flush()

        if run_test:
            for split in ["test"]:#["val", "test"]:
                # Performance without test-time augmentation
                
                # one clip
                """loss, yhat, y = ...
                f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one clip) MAPE: {:.5f} ({:.5f} - {:.5f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_percentage_error)))
                f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                f.flush()"""

                # Performance with test-time augmentation
                ds = echonet.datasets.Echo(root=data_dir, split=split, **kwargs, clips="all")
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                
                # all clips
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device, save_all=True, block_size=batch_size)
                f.write("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.r2_score)))
                f.write("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_error)))
                f.write("{} (all clips) MAPE: {:.5f} ({:.5f} - {:.5f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_percentage_error)))
                f.write("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))))
                f.flush()

                # Write full performance to file
                with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                    for (filename, pred) in zip(ds.fnames, yhat):
                        for (i, p) in enumerate(pred):
                            g.write("{},{},{:.4f}\n".format(filename, i, p))
                echonet.utils.latexify()
                yhat = np.array(list(map(lambda x: x.mean(), yhat)))

                # Plot actual and predicted EF
                plt.switch_backend('agg')
                fig = plt.figure(figsize=(3, 3))
                lower = min(y.min(), yhat.min())
                upper = max(y.max(), yhat.max())
                plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
                plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
                plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
                plt.gca().set_aspect("equal", "box")
                plt.xlabel("Actual EF (%)")
                plt.ylabel("Predicted EF (%)")
                plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split)))
                plt.close(fig)

                # Plot AUROC
                fig = plt.figure(figsize=(3, 3))
                plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
                for thresh in [35, 40, 45, 50]:
                    fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat)
                    print(thresh, sklearn.metrics.roc_auc_score(y > thresh, yhat))
                    plt.plot(fpr, tpr)

                plt.axis([-0.01, 1.01, -0.01, 1.01])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_roc.pdf".format(split)))
                plt.close(fig)


def run_epoch(model, dataloader, train, optim, device, save_all=False, block_size=None):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
        save_all (bool, optional): If True, return predictions for all
            test-time augmentations separately. If False, return only
            the mean prediction.
            Defaults to False.
        block_size (int or None, optional): Maximum number of augmentations
            to run on at the same time. Use to limit the amount of memory
            used. If None, always run on all augmentations simultaneously.
            Default is None.
    """

    model.train(train)

    total = 0  # total training loss
    n = 0      # number of videos processed
    s1 = 0     # sum of ground truth EF
    s2 = 0     # Sum of ground truth EF squared

    yhat = []
    y = []
    
    #flag = 0

    
    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, outcome) in dataloader:
                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                average = (len(X.shape) == 6)
                if average:
                    batch, n_clips, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)

                s1 += outcome.sum()
                s2 += (outcome ** 2).sum()

                if block_size is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([model(X[j:(j + block_size), ...]) for j in range(0, X.shape[0], block_size)])

                if save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                if average:
                    outputs = outputs.view(batch, n_clips, -1).mean(1)

                if not save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                loss = torch.nn.functional.mse_loss(outputs.view(-1), outcome)

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * X.size(0)
                n += X.size(0)

                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(total / n, loss.item(), s2 / n - (s1 / n) ** 2))
                pbar.update()

    if not save_all:
        yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return total / n, yhat, y

def my_run_epoch(model, mixtype, dataset_all_num, dataset, mix_alpha,batch_size, train, optim, device, mixup_idx_sample_rate=None,save_all=False, block_size=None):
    model.train(train)

    total = 0  # total training loss
    n = 0      # number of videos processed
    s1 = 0     # sum of ground truth EF
    s2 = 0     # Sum of ground truth EF squared

    yhat = []
    y = []
    
    #flag = 0
    X1, X2 = None, None
    lambd = None

    ### personal interation ###
    iteration = dataset_all_num // batch_size
    need_shuffle = 0
    all_idx = np.arange(dataset_all_num)

    # #debug
    # if train:
    #     all_idx = all_idx[:100]

    shuffle_idx = np.random.permutation(np.arange(dataset_all_num))

    is_mixup = mixtype != 'erm'

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=iteration) as pbar:
            for iter in range(iteration):
                
                #idx_1 = np.arange(len(i_list))[iter * batch_size:(iter + 1) * batch_size]
                if need_shuffle:
                    idx_1 = shuffle_idx[iter * batch_size:(iter + 1) * batch_size]
                else:
                    idx_1 = all_idx[iter * batch_size:(iter + 1) * batch_size]

                if train and is_mixup: # only train state need mixup
                    lambd = np.random.beta(mix_alpha, mix_alpha)
                    
                    if mixtype == 'random':
                        idx_2 = np.array(
                            [np.random.choice(all_idx) for sel_idx in idx_1])
                    else: # kde
                        idx_2 = np.array(
                            [np.random.choice(all_idx,
                            p = mixup_idx_sample_rate[sel_idx]) for sel_idx in idx_1])

                    X1 = np.concatenate([dataset[i][0][np.newaxis,:] for i in idx_1])
                    Y1 = np.concatenate([[dataset[i][1]] for i in idx_1])

                    X2 = np.concatenate([dataset[i][0][np.newaxis,:] for i in idx_2])
                    Y2 = np.concatenate([[dataset[i][1]] for i in idx_2])

                    
                    mixup_Y = Y1 * lambd + Y2 * (1 - lambd)
                    mixup_X = X1 * lambd + X2 * (1 - lambd)

                    outcome = mixup_Y
                    X = mixup_X

                else:
                    X       = np.concatenate([dataset[i][0][np.newaxis,:] for i in idx_1])
                    outcome = np.concatenate([[dataset[i][1]] for i in idx_1])
                    
                X = torch.tensor(X)
                outcome = torch.tensor(outcome)

                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                average = (len(X.shape) == 6)

                if average:
                    batch, n_clips, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)

                s1 += outcome.sum()
                s2 += (outcome ** 2).sum()

                if block_size is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([model(X[j:(j + block_size), ...]) for j in range(0, X.shape[0], block_size)])

                if save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                if average:
                    outputs = outputs.view(batch, n_clips, -1).mean(1)

                if not save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                loss = torch.nn.functional.mse_loss(outputs.view(-1), outcome)

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * X.size(0)
                n += X.size(0)

                pbar.set_postfix_str("total / n:{:.2f}, loss:({:.2f}) , COV/n: {:.2f}".format(total / n, loss.item(), s2 / n - (s1 / n) ** 2))
                pbar.update()

    if not save_all:
        yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return total / n, yhat, y


def get_mixup_rate(sample_list,mixtype,kde_type='gaussian',bandwidth=1.0,device='cuda'):
    mix_idx = []
    if len(sample_list.shape) == 1:
        sample_list = sample_list[:,np.newaxis]
    with tqdm.tqdm(total=len(sample_list)) as pbar:
        #pbar.set_description('Cal Sample Rate ')
        for i in range(len(sample_list)):
            if mixtype == 'kde': 
                data_i = sample_list[i]
                #xi = x_list[i]
                data_i = data_i[:,np.newaxis]
                kd = KernelDensity(kernel=kde_type, bandwidth=bandwidth).fit(data_i)
                each_rate = np.exp(kd.score_samples(sample_list))
                each_rate /= np.sum(each_rate)  # norm
            else: # random
                each_rate = np.ones(sample_list.shape[0]) * 1.0 / sample_list.shape[0]
            
            mix_idx.append(each_rate)

            pbar.set_postfix_str('rate: max: {:.5f}, min: {:.8f}, std: {:.5f}, mean: {:.5f}'.format(max(each_rate),min(each_rate),np.std(each_rate),np.mean(each_rate)))

    return np.array(mix_idx)


def stats_values(targets, flag = False):
    mean = torch.mean(targets)
    min = torch.min(targets)
    max = torch.max(targets)
    std = torch.std(targets)
    if flag:
        print(f'y stats: mean = {mean}, max = {max}, min = {min}, std = {std}')
    return mean, min, max, std