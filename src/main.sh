# airfoil 
python main.py --dataset Airfoil --mixtype kde --kde_bandwidth 1.75 --use_manifold 1 --store_model 1 --read_best_model 0

# no2 
python main.py --dataset NO2 --mixtype kde --kde_bandwidth 1.2 --use_manifold 0 --store_model 1 --read_best_model 0

# exchange_rate 
python main.py --dataset TimeSeries --data_dir ./data/exchange_rate/exchange_rate.txt --ts_name exchange_rate --mixtype kde --kde_bandwidth 5e-2 --use_manifold 1 --store_model 1 --read_best_model 0

# electricity
python main.py --dataset TimeSeries --data_dir ./data/electricity/electricity.txt --ts_name electricity --mixtype kde --kde_bandwidth 0.5 --use_manifold 0 --store_model 1 --read_best_model 0

# rcf-mnist
python main.py --dataset RCF_MNIST --data_dir ./data/RCF_MNIST --mixtype random --batch_type 1 --kde_bandwidth 0.2 --use_manifold 1 --store_model 1 --read_best_model 0

# crime
python main.py --dataset CommunitiesAndCrime --mixtype kde --kde_bandwidth 4.0 --use_manifold 1 --store_model 1 --read_best_model 0

# skillcraft
python main.py --dataset SkillCraft --mixtype kde --kde_bandwidth 1.0 --use_manifold 0 --store_model 1 --read_best_model 0

# dti_dg
python main.py --dataset Dti_dg --data_dir ../../dti_dg/domainbed/data/ --mixtype kde --kde_bandwidth 10.0 --use_manifold 1 --store_model 1 --read_best_model 0