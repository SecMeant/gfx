#!/bin/python3
import argparse

def NumberList(value):
    if not ',' in value:
        return [value]
    return [int(x) for x in value.split(',')]

def NumberList2(value):
    l = NumberList(value)
    if len(l) != 2 or l[0] > l[1]:
        raise argparse.ArgumentError(None, "Expected list of two values: min,max.")
    return l

parser = argparse.ArgumentParser(
        description = 'Generates random matricies of given dimension and computes the product')

parser.add_argument('-o', '--out', required = True, type = str)
parser.add_argument('-d', '--dim', required = True, type = int)

text = 'Value range in form of "min,max".'
parser.add_argument('-v', '--valrange', required = True, type = NumberList2, help = text)

text = 'Number of matrix products'
parser.add_argument('-n', '--repeat', dest = 'n', default = 32, required = True, type = int, help = text)

parser.add_argument('-t', '--type', choices = ['i32', 'f32'], default = 'i64', type = str)

parser.add_argument('--verbose', action = 'store_true', default = False)

args = parser.parse_args()

import torch
from safetensors.torch import load_file, save_file

if args.type == 'i32':
    dtype = torch.i32
elif args.type == 'f32':
    dtype = torch.float32

dim = (args.dim, args.dim)
tensors = {}

for i in range(args.n):
    A = torch.randint(args.valrange[0], args.valrange[1], dim, dtype=dtype)
    B = torch.randint(args.valrange[0], args.valrange[1], dim, dtype=dtype)
    C = A @ B

    tensors[f'A{i}'] = A
    tensors[f'B{i}'] = B
    tensors[f'C{i}'] = C

    if args.verbose:
        print(f'{A=}\n{B=}\n{C=}\n')

save_file(tensors, args.out)

