
import data.airfoil as airfoil
import data.no2 as no2
import data.time_series as ts
import data.communities_and_crime as cc
import data.skillcraft as sc
import data.dti_dg as dd
import data.RCF_MNIST as rfm

import pickle


def load_data(args, device = 'cuda'):
    data_packet, ts_data = None, None
    if args.dataset == 'Airfoil':
        assert args.is_ood == 0
        data_packet = airfoil.get_Airfoil_data_packet(args,f'./data/{args.dataset}/')
    elif args.dataset == 'NO2':
        assert args.is_ood == 0
        data_packet = no2.get_NO2_data_packet(args,f'./data/{args.dataset}/')
    elif args.dataset == 'TimeSeries':
        assert args.is_ood == 0
        use_cuda = 'cuda' in device
        ts_data = ts.Data_utility(args.data_dir, 0.6, 0.2, 
                                use_cuda, args.horizon, args.window, args.normalize, device) #0.6 0.2
        data_packet = ts.get_TimeSeries_data_packet(args, ts_data)

    elif args.dataset == 'CommunitiesAndCrime':
        data_packet = cc.get_CommunitiesAndCrime_data_packet(args,f'./data/{args.dataset}/')

    elif args.dataset == 'SkillCraft':
        data_packet = sc.get_SkillCraft_data_packet(args,f'./data/{args.dataset}/')

    elif args.dataset == 'Dti_dg':
        dd_hparams = dd.get_hparams(args)
        args.batch_size = dd_hparams['batch_size']
        args.lr = dd_hparams['lr']
        data_packet = dd.get_Dti_dg_data_packet(args,dd_hparams)
    elif args.dataset == 'RCF_MNIST':
        data_packet = rfm.get_RCF_MNIST_data_packet(args)
    
    if args.show_setting:
        print(
            f"x.tr,va,te; y.tr,va,te.shape = {data_packet['x_train'].shape, data_packet['x_valid'].shape, data_packet['x_test'].shape, data_packet['y_train'].shape, data_packet['y_valid'].shape, data_packet['y_test'].shape}"+
            f"y.tr.mean = {data_packet['y_train'].mean()}, y.tr.std = {data_packet['y_train'].std()}")

    return data_packet, ts_data

