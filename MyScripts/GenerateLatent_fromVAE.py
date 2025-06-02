import torch
import numpy as np

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.vae_adain import Model as VAE
from default_config import cfg

# Path to ckpts
#ckpt_path = "/home/ncaytuir/data-local/exp/0513/airplane/28c7bch_hvae_lion_B16/checkpoints/epoch_7999_iters_159999.pt"
#cfg_path = "/home/ncaytuir/data-local/exp/0513/airplane/28c7bch_hvae_lion_B16/cfg.yml"
ckpt_path = "/home/ncaytuir/data-local/exp/0523 (3th VAE 5)/airplane/28c7bch_hvae_lion_B16/checkpoints/epoch_7999_iters_159999.pt"
cfg_path = "/home/ncaytuir/data-local/exp/0523 (3th VAE 5)/airplane/28c7bch_hvae_lion_B16/cfg.yml"

real_shape = "/home/ncaytuir/data-local/LION_necs/output/pc_0.npy"

ouput_path = "/home/ncaytuir/data-local/LION_necs/output/latent_test04.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg.merge_from_file(cfg_path)
args = cfg

# Load ckpts
ckpt = torch.load(ckpt_path, map_location=device)

print("args", args)

# Instanciate model
vae = VAE(args).to(device)
vae.load_state_dict(ckpt["model"])
vae.eval()

""" # Params
num_samples = 1
num_points = vae.num_points
input_dim = vae.input_dim
latent_dim = vae.latent_dim
latent_dim_ext = args.latent_pts.latent_dim_ext[0] if hasattr(args.latent_pts, 'latent_dim_ext') else 0

print("num_points", num_points)
print("input_dim", input_dim)
print("latent_dim", latent_dim)
print("latent_dim_ext", latent_dim_ext)

# Sample z_local from noise
z_local = torch.randn(num_samples, num_points * (latent_dim + input_dim)).to(device)

# Covert to latent h_0
latent_shape = [num_samples, num_points, latent_dim + input_dim]
h_0 = z_local.view(*latent_shape)[:, :, :input_dim]

np.save(ouput_path, h_0.squeeze(0).cpu().numpy())
print("Done") """

# Load real cloud
x = np.load(real_shape)
if x.ndim == 2:
    x = x[None, ...]
x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

# Obtain h_0
with torch.no_grad():
    h0 = vae.get_latent_points(x_tensor)

np.save(ouput_path, h0.squeeze(0).cpu().numpy())
print("done")