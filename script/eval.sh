#python train_dist.py --skip_nll 1 --eval_generation --pretrained  $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3 # 3
#python train_dist.py --skip_nll 1 --eval_generation --pretrained $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3

# VAE
#python train_dist.py --skip_nll 1 --eval_generation --pretrained $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3

# PRIOR
#python train_dist.py --skip_nll 1 --eval_generation --pretrained $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3 

# Just generate

#checkpoint="/home/ncaytuir/data-local/exp/0604/airplane/7807deh_train_lion_B10/checkpoints/epoch_17999_iters_575999.pt" 

#python train_dist.py --skip_nll 1 --eval_generation --pretrained $1 --skip_sample 0 --ntest 1000

#python train_dist.py --skip_nll 1 --skip_sample 0 --eval_generation --pretrained $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3 

#checkpoint="/home/ncaytuir/data-local/exp/1006/airplane/824c5bh_train_lion_B10/checkpoints/epoch_5999_iters_563999.pt"

#python train_dist.py --skip_nll 1 --eval_generation --pretrained  $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3 # 3
#python train_dist.py --skip_nll 1 --eval_generation --pretrained $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3

# VAE
#python train_dist.py --skip_nll 1 --eval_generation --pretrained $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3

# PRIOR
#python train_dist.py --skip_nll 1 --eval_generation --pretrained $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3 

# Just generate

#checkpoint="/home/ncaytuir/data-local/exp/0604/airplane/7807deh_train_lion_B10/checkpoints/epoch_17999_iters_575999.pt" 

#python train_dist.py --skip_nll 1 --eval_generation --pretrained $1 --skip_sample 0 --ntest 1000

#python train_dist.py --skip_nll 1 --skip_sample 0 --eval_generation --pretrained $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3 

#checkpoint="/home/ncaytuir/data-local/exp/1006/airplane/824c5bh_train_lion_B10/checkpoints/epoch_5999_iters_563999.pt"

# Prior
# Airplane
python train_dist.py --skip_nll 1 --eval_generation --pretrained /home/ncaytuir/exp/patagon_priors/airplane/checkpoints/epoch_3999_iters_375999.pt ddpm.model_var_type "fixedlarge" data.batch_size_test 200 ddpm.ema 1 num_val_samples 3

python train_dist.py --skip_nll 1 --eval_generation --pretrained /home/ncaytuir/exp/patagon_priors/airplane/checkpoints/epoch_6999_iters_657999.pt ddpm.model_var_type "fixedlarge" data.batch_size_test 200 ddpm.ema 1 num_val_samples 3

python train_dist.py --skip_nll 1 --eval_generation --pretrained /home/ncaytuir/exp/patagon_priors/airplane/checkpoints/epoch_10999_iters_1033999.pt ddpm.model_var_type "fixedlarge" data.batch_size_test 200 ddpm.ema 1 num_val_samples 3

python train_dist.py --skip_nll 1 --eval_generation --pretrained /home/ncaytuir/exp/patagon_priors/airplane/checkpoints/epoch_14999_iters_1409999.pt ddpm.model_var_type "fixedlarge" data.batch_size_test 200 ddpm.ema 1 num_val_samples 3

# Car
python train_dist.py --skip_nll 1 --eval_generation --pretrained /home/ncaytuir/exp/patagon_priors/car/checkpoints/epoch_3999_iters_195999.pt ddpm.model_var_type "fixedlarge" data.batch_size_test 200 ddpm.ema 1 num_val_samples 3

python train_dist.py --skip_nll 1 --eval_generation --pretrained /home/ncaytuir/exp/patagon_priors/car/checkpoints/epoch_6999_iters_342999.pt ddpm.model_var_type "fixedlarge" data.batch_size_test 200 ddpm.ema 1 num_val_samples 3

python train_dist.py --skip_nll 1 --eval_generation --pretrained /home/ncaytuir/exp/patagon_priors/car/checkpoints/epoch_10999_iters_538999.pt ddpm.model_var_type "fixedlarge" data.batch_size_test 200 ddpm.ema 1 num_val_samples 3

python train_dist.py --skip_nll 1 --eval_generation --pretrained /home/ncaytuir/exp/patagon_priors/car/checkpoints/epoch_14999_iters_734999.pt ddpm.model_var_type "fixedlarge" data.batch_size_test 200 ddpm.ema 1 num_val_samples 3

# Chair
python train_dist.py --skip_nll 1 --eval_generation --pretrained /home/ncaytuir/exp/patagon_priors/chair/checkpoints/epoch_3999_iters_367999.pt ddpm.model_var_type "fixedlarge" data.batch_size_test 200 ddpm.ema 1 num_val_samples 3

python train_dist.py --skip_nll 1 --eval_generation --pretrained /home/ncaytuir/exp/patagon_priors/chair/checkpoints/epoch_6999_iters_643999.pt ddpm.model_var_type "fixedlarge" data.batch_size_test 200 ddpm.ema 1 num_val_samples 3

python train_dist.py --skip_nll 1 --eval_generation --pretrained /home/ncaytuir/exp/patagon_priors/chair/checkpoints/epoch_10999_iters_1011999.pt ddpm.model_var_type "fixedlarge" data.batch_size_test 200 ddpm.ema 1 num_val_samples 3

python train_dist.py --skip_nll 1 --eval_generation --pretrained /home/ncaytuir/exp/patagon_priors/chair/checkpoints/epoch_14999_iters_1379999.pt ddpm.model_var_type "fixedlarge" data.batch_size_test 200 ddpm.ema 1 num_val_samples 3