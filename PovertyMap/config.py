
dataset_defaults = {
    'poverty': {
        'epochs': 50,
        'batch_size': 64,
        'optimiser': 'Adam',
        'optimiser_args': { 
            'lr': 1e-3,
            'weight_decay': 0,
            #'amsgrad': True,
            #'betas': (0.9, 0.999),
        },
        'pretrain_iters': 0,
        'meta_lr': 0.1,
        'meta_steps': 5,
        'selection_metric': 'r_wg',
        'reload_inner_optim': True,
        'print_iters': 350,
        'scheduler': 'StepLR',
        'scheduler_kwargs': {'gamma': 0.96},
    }
}

