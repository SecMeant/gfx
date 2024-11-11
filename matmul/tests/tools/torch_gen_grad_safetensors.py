#!/bin/python3

#
# Parse arguments
#

import argparse

def IntList(value):
    if not ',' in value:
        return [int(value)]
    return [int(x) for x in value.split(',')]

def IntList2(value):
    l = IntList(value)
    if len(l) != 2 or l[0] > l[1]:
        raise argparse.ArgumentError(None, "Expected list of two values: min,max.")
    return l

parser = argparse.ArgumentParser(
        description = 'Generates random input vector x and output vector y')

parser.add_argument('-o', '--out', required = True, type = str, help = 'Output file')

text = 'Input and output vector dimensions.'
parser.add_argument('-d', '--dim', required = True, type = IntList, help = text)

text = 'Range of values, given as "min,max" that we fill the matricies with.'
parser.add_argument('-v', '--valrange', required = True, type = IntList2, help = text)

args = parser.parse_args()


#
# Generate safetensors
#

import torch
from safetensors.torch import save_file

val_range = (args.valrange[0], args.valrange[1])
dtype = torch.float32
tensors = {}

tensors['x'] = torch.randint(val_range[0], val_range[1], args.dim, dtype=dtype)
tensors['y'] = torch.randint(val_range[0], val_range[1], args.dim, dtype=dtype)

save_file(tensors, args.out)