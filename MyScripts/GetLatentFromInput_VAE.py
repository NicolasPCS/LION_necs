import os
import torch
import numpy as np
import time
from default_config import cfg as config
from models.lion import LION
from models.vae_adain import Model as VAE

# Paths
model_path = ""
model_config = ""
input_dir = ""
output_dir = ""

# Config
config.merge_from_file(model_config)
print(model_config)
#LION
lion = LION(config)
lion.load_model(model_path)

# VAE
vae = VAE(config).cuda()
vae.eval()
# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def latent_points():
    # Iterate over all .npy files in the directory
    with torch.no_grad(): # Disable gradient tracking
        for filename in os.listdir(input_dir):
            if filename.endswith('.npy'):
                # Load the point cloud data from the .npy file
                file_path = os.path.join(input_dir, filename)
                point_cloud = np.load(file_path)

                # Add a batch dimension if needed
                point_cloud = np.expand_dims(point_cloud, axis = 0) # Now shape is [1, N, point_dim]

                # Convert to Pytorch tensor and move to device
                point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float32).to(device)

                # Get the latent representations
                h0 = vae.get_latent_points(point_cloud_tensor)

                torch.cuda.synchronize()

                # Save the latent points with a name based on the original file
                output_filename = f'latent_points_{os.path.splitext(filename)[0]}.pth'
                torch.save(h0.cpu(), os.path.join(output_dir, output_filename)) # Save on CPU to free GPU memory
                print(f"Saved {output_filename}!")
                time.sleep(0.1)
                torch.cuda.empty_cache()

latent_points()

def lion_sample_no_decode():
    output_dict = lion.sample()
    o_class = 'airplane'

    z_reshape = output_dict['z_local'].view(1, 2048, 4,1,1)
    h0 = z_reshape[:,:,:3] # Slice to get the first 3 channels (x, y, z)
    h0 = h0.detach().cpu().view(1, -1, 3) # Detach, move to CPU and reshape
    output_filename = f'latent_point_{o_class}_z_local.pth'
    torch.save(h0, os.path.join(output_dir, output_filename))
    print(f"Saved {output_filename}!")