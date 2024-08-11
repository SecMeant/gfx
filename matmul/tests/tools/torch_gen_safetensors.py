#!/bin/python3
import argparse

parser = argparse.ArgumentParser(
        description = 'Generates random matricies of given dimension and computes the product')

parser.add_argument('-o', '--out', required = True, type = str)
parser.add_argument('-d', '--dim', required = True, type = int)
parser.add_argument('-v', '--maxval', required = True, type = int)
parser.add_argument('-n', '--repeat', dest = 'n', default = 32, required = True, type = int,
                    help = "Number of matrix products")

args = parser.parse_args()

import torch
from safetensors.torch import load_file, save_file

dim = (args.dim, args.dim)
val_range = (0, args.maxval)
dtype = torch.int32
tensors = {}

for i in range(args.n):
    A = torch.randint(val_range[0], val_range[1], dim, dtype=dtype)
    B = torch.randint(val_range[0], val_range[1], dim, dtype=dtype)
    C = A @ B

    tensors[f'A{i}'] = A
    tensors[f'B{i}'] = B
    tensors[f'C{i}'] = C

save_file(tensors, args.out)

