# C-Mixup: Improving Generalization in Regression

Official code of C-Mixup.


If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{yao2022cmix,
  title={C-Mixup: Improving Generalization in Regression},
  author={Yao, Huaxiu and Wang, Yiping and Zhang, Linjun and Zou, James and Finn, Chelsea},
  booktitle={Proceeding of the Thirty-Sixth Conference on Neural Information Processing Systems},
  year={2022}
}
```


## Prerequisites
- python 3.7.13
- matplotlib 3.3.4
- numpy 1.20.1
- pandas 1.2.3
- pillow 9.0.1
- pytorch 1.11.0
- pytorch_transformers 1.2.0
- torchvision 0.9.0
- wilds 2.0.0

## Datasets and Scripts

We put all code except Echo and PovertyMap on the `src` folder. Echo and PovertyMap datasets are built upon different codebase, which are put in the `echo` and `povertymap` folders, respectively. 


### Airfoil
This dataset can be downloaded via the link in the [Google Drive](https://drive.google.com/drive/folders/1pTRT7fA-hq6p1F7ZX5oJ0tg_I1RRG6OW?usp=sharing). Please put the corresponding datafolder to `src/data`


The command to run C-Mixup on Airfoil is:
```
python main.py --dataset Airfoil --mixtype kde --kde_bandwidth 1.75 --use_manifold 1 --store_model 1 --read_best_model 0
```

### NO2 
This dataset can be downloaded via the link in the [Google Drive](https://drive.google.com/drive/folders/1pTRT7fA-hq6p1F7ZX5oJ0tg_I1RRG6OW?usp=sharing). Please put the corresponding datafolder to `src/data`


The command to run C-Mixup on NO2 is:
```
python main.py --dataset NO2 --mixtype kde --kde_bandwidth 1.2 --use_manifold 0 --store_model 1 --read_best_model 0
```

### Exchange_rate
This dataset can be downloaded via the link in the [Google Drive](https://drive.google.com/drive/folders/1pTRT7fA-hq6p1F7ZX5oJ0tg_I1RRG6OW?usp=sharing). Please put the corresponding datafolder to `src/data`


The command to run C-Mixup on Exchange_rate is:
```
python main.py --dataset TimeSeries --data_dir ./data/exchange_rate/exchange_rate.txt --ts_name exchange_rate --mixtype kde --kde_bandwidth 5e-2 --use_manifold 1 --store_model 1 --read_best_model 0
```

### Electricity

This dataset can be downloaded via the link in the [Google Drive](https://drive.google.com/drive/folders/1pTRT7fA-hq6p1F7ZX5oJ0tg_I1RRG6OW?usp=sharing). Please put the corresponding datafolder to `src/data`


The command to run C-Mixup on Electricity is:
```
python main.py --dataset TimeSeries --data_dir ./data/electricity/electricity.txt --ts_name electricity --mixtype kde --kde_bandwidth 0.5 --use_manifold 0 --store_model 1 --read_best_model 0
```

### RCF-MNIST

This dataset can be downloaded via the link in the [Google Drive](https://drive.google.com/drive/folders/1pTRT7fA-hq6p1F7ZX5oJ0tg_I1RRG6OW?usp=sharing). Please put the corresponding datafolder to `src/data`


The command to run C-Mixup on RCF-MNIST is:
```
python main.py --dataset RCF_MNIST --data_dir ./data/RCF_MNIST --mixtype random --batch_type 1 --kde_bandwidth 0.2 --use_manifold 1 --store_model 1 --read_best_model 0
```

### Crime
This dataset can be downloaded via the link in the [Google Drive](https://drive.google.com/drive/folders/1pTRT7fA-hq6p1F7ZX5oJ0tg_I1RRG6OW?usp=sharing). Please put the corresponding datafolder to `src/data`


The command to run C-Mixup on Crime is:
```
python main.py --dataset CommunitiesAndCrime --mixtype kde --kde_bandwidth 4.0 --use_manifold 1 --store_model 1 --read_best_model 0
```

### Skillcraft
This dataset can be downloaded via the link in the [Google Drive](https://drive.google.com/drive/folders/1pTRT7fA-hq6p1F7ZX5oJ0tg_I1RRG6OW?usp=sharing). Please put the corresponding datafolder to `src/data`


The command to run C-Mixup on Skillcraft is:
```
python main.py --dataset SkillCraft --mixtype kde --kde_bandwidth 1.0 --use_manifold 0 --store_model 1 --read_best_model 0
```

### DTI
This dataset can be downloaded via the link in the [Google Drive](https://drive.google.com/drive/folders/1pTRT7fA-hq6p1F7ZX5oJ0tg_I1RRG6OW?usp=sharing). Please put the corresponding datafolder to `src/data`


The command to run C-Mixup on DTI is:
```
python main.py --dataset Dti_dg --data_dir ./data/dti --mixtype kde --kde_bandwidth 20.0 --use_manifold 1 --store_model 1 --read_best_model 0
```

### PovertyMap

To get detailed information of the datasets, please refer to Appendix E of the [paper](https://arxiv.org/abs/2210.05775v1) or [original paper](https://arxiv.org/abs/2112.05090).

This code is built upon [LISA](https://github.com/huaxiuyao/LISA) and [Wilds](https://github.com/p-lambda/wilds).

Before running, please `cd PovertyMap`

The datasets will be automatically downloaded when running the scripts provided below.

```
python main.py --dataset poverty --algorithm mixup --data-dir ../../datasets/ --experiment_dir .. --is_kde 1 --kde_bandwidth 0.5
```


### EchoNet

To get detailed information of the datasets, please refer to the [website](https://echonet.github.io/dynamic/).

This code is built upon [EchoNet](https://github.com/echonet/dynamic).

Before running, please `cd EchoNet`.

You need to follow the guideline from the [website](https://echonet.github.io/dynamic/index.html#access) and download the dataset into `../../EchoNet-Dynamic/` directory first.

For the preparation you need to install the echonet environment and complete segmentation tasks by running the commands:
```
pip install --upgrade --user . 
python echonet/__main__.py segmentation --save_video
```

The command to run C-Mixup on EchoNet is:
```
echonet video --batch_size 10 --device cuda --num_workers 0 --num_epochs 20 --mixtype kde --bandwidth 50.0 --run_test True
```