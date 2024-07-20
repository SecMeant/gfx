#!/bin/python3
import torch
from safetensors.torch import load_file, save_file
from sys import argv

if len(argv) <= 1:
    out_path = 'pytorch_model.safetensors'
else:
    out_path = argv[1]

dim = (128, 128)
val_range = (0, 4096)
dtype = torch.int32
tensors = {}

for i in range(32):
    A = torch.randint(val_range[0], val_range[1], dim, dtype=dtype)
    B = torch.randint(val_range[0], val_range[1], dim, dtype=dtype)
    C = A @ B

    tensors[f'A{i}'] = A
    tensors[f'B{i}'] = B
    tensors[f'C{i}'] = C

save_file(tensors, out_path)

