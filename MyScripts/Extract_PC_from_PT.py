import os
import torch
import numpy as np

data = torch.load("/home/ncaytuir/data-local/LION_necs/datasets/test_data/ref_val_airplane.pt", map_location='cpu')
output_dir = "/home/ncaytuir/data-local/LION_necs/datasets/test_data/pcs_airplane"

#print(type(data))
#points = data if isinstance(data, torch.Tensor) else data['points']
#points = data.numpy()
#print(points[0])

j = 0

data = data['ref']

for i in range(data.shape[0]):
    np.save(os.path.join(output_dir, f"pc_{i}.npy"), data[i].numpy())
    j += 1

print("Done")
print(j)