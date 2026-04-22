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
python train_dist.py --skip_nll 1 --eval_generation --pretrained /home/isipiran/exp/0411/airplane/9e2642h_train_lion_B10/checkpoints/epoch_17999_iters_1691999.pt ddpm.model_var_type "fixedlarge" data.batch_size_test 200 ddpm.ema 1 num_val_samples 3

python train_dist.py --skip_nll 1 --eval_generation --pretrained /home/isipiran/exp/0406/car/f64befh_train_lion_B10/checkpoints/epoch_17999_iters_881999.pt ddpm.model_var_type "fixedlarge" data.batch_size_test 200 ddpm.ema 1 num_val_samples 3

python train_dist.py --skip_nll 1 --eval_generation --pretrained /home/isipiran/exp/0412/chair/7d8498h_train_lion_B10/checkpoints/epoch_17999_iters_1655999.pt ddpm.model_var_type "fixedlarge" data.batch_size_test 200 ddpm.ema 1 num_val_samples 3