dataset_defaults = {
    'Airfoil':{
        'input_dim': 5,
        'batch_size': 16,
        'num_epochs': 100,
        'optimiser_args': {
            'lr': 1e-2,
        },
        'metrics': 'rmse',
        'is_ood': 0,
        # split
        'train_ratio':0.7,
        'valid_ratio':0.1,
        'id_train_val_split':[0.7, 0.1, 0.2],
        # mixup
        'mix_alpha': 2.0
    },
    'NO2':{
        'input_dim': 7,
        'batch_size': 32,
        'num_epochs': 100,
        'optimiser_args': {
            'lr': 1e-2,
        },
        'metrics': 'rmse',
        'is_ood': 0,
        # split
        'train_ratio':0.7,
        'valid_ratio':0.1,
        'id_train_val_split':[0.7, 0.1, 0.2],
        # mixup
        'mix_alpha': 2.0
    },
    'TimeSeries-exchange_rate':{ # ref -> LSTNet
        'batch_size': 128,
        'num_epochs': 100,
        'optimiser_args': {
            'lr': 1e-3,
        },
        'metrics': 'rmse',
        'is_ood': 0,
        # split
        'train_ratio':0.7,
        'valid_ratio':0.1,
        'id_train_val_split':[0.7, 0.1, 0.2],
        # mixup
        'mix_alpha': 1.5,
        # ts
        'hidCNN': 50, # number of CNN hidden units
        'hidRNN': 50, # number of RNN hidden units
        'window': 24*7, # window size
        'CNN_kernel': 6, # the kernel size of the CNN layers
        'highway_window': 24, # The window size of the highway component
        'clip': 10., # gradient clipping
        'dropout': 0.2, # dropout applied to layers (0 = no dropout)
        'horizon': 12, 
        'skip': 24,
        'hidSkip': 5,
        'L1Loss':False,
        'normalize': 2,
        'output_fun':None,
    },

    'TimeSeries-electricity':{
        'batch_size': 128,
        'num_epochs': 100,
        'optimiser_args': {
            'lr': 1e-3,
        },
        'metrics': 'rmse',
        'is_ood': 0,
        # split
        'train_ratio':0.7,
        'valid_ratio':0.1,
        'id_train_val_split':[0.7, 0.1, 0.2],
        # mixup
        'mix_alpha': 2.0,
        # ts
        'hidCNN': 100,
        'hidRNN': 100,
        'window': 24*7,
        'CNN_kernel': 6,
        'highway_window': 24,
        'clip': 10.,
        'dropout': 0.2,
        'horizon': 24,
        'skip': 24,
        'hidSkip': 5,
        'L1Loss':False,
        'normalize': 2,
        'output_fun':'Linear',
    },

    'RCF_MNIST':{
        'batch_size': 64,
        'num_epochs': 30,
        'optimiser_args': {
            'lr': 7e-5,
        },
        'metrics': 'rmse',
        'is_ood': 1,
        # split
        'train_ratio':0.7,
        'valid_ratio':0.1,
        'id_train_val_split':[0.7, 0.1, 0.2],
        # mixup
        'mix_alpha': 2.0,
        # rcf
        'use_rotate_class': 0, # 0 -> generate degree randomly; 1 -> sample from 60 fix degree levels
        'spurious_ratio': 0.8,
        'construct_color_data': -1, # 1 -> construct new r-mnist with color spurious information; 0 -> do nothing; -1 -> read rc-fmnist
        'construct_no_color_data': 0, # 1 -> construct r-fmnist; 0 -> do nothing; -1 -> read r-fmnist
        'vis_rcf': 0, # visualize generated data
        'all_pos_color': 0, # 1 -> test data has inverse spurious feature; 0 -> test data has spurious feature as train data
    },

    'CommunitiesAndCrime':{
        'input_dim': 122,
        'batch_size': 16,
        'num_epochs': 200,
        'optimiser_args': {
            'lr': 1e-3,
        },
        'metrics': 'rmse',
        'is_ood': 1,
        # split
        'train_ratio':0.7,
        'valid_ratio':0.1,
        'id_train_val_split':[0.7, 0.1, 0.2],
        # mixup
        'mix_alpha': 2.0
    },

    'SkillCraft':{
        'input_dim': 17,
        'batch_size': 32,
        'num_epochs': 100,
        'optimiser_args': {
            'lr': 1e-2,
        },
        'metrics': 'rmse',
        'is_ood': 1,
        # split
        'train_ratio':0.7,
        'valid_ratio':0.1,
        'id_train_val_split':[0.7, 0.1, 0.2],
        # mixup
        'mix_alpha': 2.0
    },

    'Dti_dg':{
        'batch_size': 64,
        'num_epochs': 20,
        'optimiser_args': {
            'lr': 5e-5,
        },
        'metrics': 'r',
        'is_ood': 1,
        # split
        'train_ratio':0.7,
        'valid_ratio':0.1,
        'id_train_val_split':[0.7, 0.1, 0.2],
        # mixup
        'mix_alpha': 2.0,
        # dti
        'read_dataset': 1, # input dataset directly
        'sub_sample_batch_max_num': 100,
        'store_log': 0,
        'task':'domain_generalization',
        'algorithm':'ERM',
        'hparams': '',
        'hparams_seed': 0,
        'trial_seed': 0,
        'test_envs': [6, 7, 8],
        'output_dir': 'train_output',
        'holdout_fraction': 0.2,
        'uda_holdout_fraction': 0., # For domain adaptation, % of test to use unlabeled for training
    },
}
