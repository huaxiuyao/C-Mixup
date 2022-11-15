pip install --upgrade --user . 
# seg
python echonet/__main__.py segmentation --save_video
# video
# erm
echonet video --batch_size 10 --device cuda --num_workers 0 --num_epochs 20 --mixtype erm --run_test True

# random
echonet video --batch_size 10 --device cuda --num_workers 0 --num_epochs 20 --mixtype random --run_test True

# kde
echonet video --batch_size 10 --device cuda --num_workers 0 --num_epochs 20 --mixtype kde --bandwidth 50.0 --run_test True
