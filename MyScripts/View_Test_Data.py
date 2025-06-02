import torch
import numpy as np

tensor1 = torch.load('/home/ncaytuir/data-local/LION_necs/datasets/test_data/ref_val_airplane.pt')
tensor2 = torch.load('/home/ncaytuir/data-local/LION_necs/datasets/test_data/ref_val_car.pt')
tensor3 = torch.load('/home/ncaytuir/data-local/LION_necs/datasets/test_data/ref_val_chair.pt')

print(tensor1['ref'].shape)
print(tensor2['ref'].shape)
print(tensor3['ref'].shape)