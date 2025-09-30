#python train_dist.py --skip_nll 1 --eval_generation --pretrained  $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3 # 3
#python train_dist.py --skip_nll 1 --eval_generation --pretrained $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3

# VAE
#python train_dist.py --skip_nll 1 --eval_generation --pretrained $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3

# PRIOR
#python train_dist.py --skip_nll 1 --eval_generation --pretrained $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3 

# Just generate

#checkpoint="/home/ncaytuir/data-local/exp/0604/airplane/7807deh_train_lion_B10/checkpoints/epoch_17999_iters_575999.pt" 

#python train_dist.py --skip_nll 1 --eval_generation --pretrained $1 --skip_sample 0 --ntest 1000

python train_dist.py --skip_nll 1 --skip_sample 0 --eval_generation --pretrained $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3 