#!/bin/python
#
# Generates safetensors test files
#

from subprocess import call as run_program, DEVNULL
from argparse import ArgumentParser
from os import makedirs,path

parser = ArgumentParser(description='Generate safetensor files for testing.')

parser.add_argument(
    '-o', '--outdir',
    type=str,
    required=True,
    help='Path to the output directory'
)

args = parser.parse_args()

tools_path = path.normpath(path.dirname(__file__))
makedirs(args.outdir, exist_ok = True)

#
# Generate classify-moons.safetensors
#
program_args = [
    path.join(tools_path,'gen-classify-moons.py'),
    '-o', path.join(args.outdir, 'classify-moons.safetensors'),
]
print('RUN ', ' '.join(program_args))
run_program(program_args, stdout = DEVNULL, stderr = DEVNULL)


#
# Generate classify.safetensors
#
program_args = [
    path.join(tools_path, 'gen-classify.py'),
    '-o', path.join(args.outdir, 'classify.safetensors'),
]
print('RUN ', ' '.join(program_args))
run_program(program_args)


#
# Generate grad.safetensors
#
valrange = (-16, 16)
dim = (4,1)
program_args = [
    path.join(tools_path, 'torch_gen_grad_safetensors.py'),
    '-o', path.join(args.outdir, 'grad.safetensors'),

    # Ugly hack is needed for arg to be ' dim,dim' and ' valrange,valrange',
    # so that argparse does't interpret comma incorrectly.
    '-d', f' {dim[0]},{dim[1]}',
    '-v', f' {valrange[0]},{valrange[1]}',
]
print('RUN ', ' '.join(program_args))
run_program(program_args)


#
# Generate pytorch_2048x2024_f32.safetensors and alike files
#
datatypes = ['i32', 'f32']
dims = [2048, 1024, 512, 256, 128, 64, 4]
prefix = 'pytorch_'
suffix = '.safetensors'
for dt in datatypes:
    for dim in dims:
        valrange = (0, 16)
        repeat = 4

        dt_str = ''
        if dt == 'f32':
            dt_str = '_f32'

        program_args = [
            path.join(tools_path, 'torch_gen_safetensors.py'),
            '-o', path.join(args.outdir, f'{prefix}{dim}x{dim}{dt_str}{suffix}'),
            '-d', f'{dim}',
            '-v', f'{valrange[0]},{valrange[1]}',
            '-n', f'{repeat}', 
            '-t', dt
        ]
        print('RUN ', ' '.join(program_args))
        run_program(program_args)
