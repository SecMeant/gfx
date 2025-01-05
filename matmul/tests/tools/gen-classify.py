#!/bin/python3
# vim: set textwidth=120 shiftwidth=4 softtabstop=4 expandtab:

import argparse

parser = argparse.ArgumentParser(
    description = "Generates 4 sets of points belonging to 2 classes")

parser.add_argument('-o', '--out', required = True, type = str, help = "Output file")
parser.add_argument('-s', '--show', default = False, action = 'store_true', help = "Show/visualise generated data")

args = parser.parse_args()


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class Circle:
    def __init__(self, x, y, r, c):
        self.x = float(x)
        self.y = float(y)
        self.r = float(r)
        self.c = c
        self.points = []

class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

def generate_point_in_circle(circle):
    angle = np.random.uniform(0., 2. * np.pi)

    r = np.random.uniform(0., circle.r)

    x = circle.x + r * np.cos(angle)
    y = circle.y + r * np.sin(angle)

    return Point(x, y)

# Generate random data
np.random.seed(31337)  # For reproducibility

CLASS_BLUE = 0
CLASS_RED  = 1

def class2color(c):
    if c == CLASS_BLUE:
        return 'blue'

    if c == CLASS_RED:
        return 'red'

    return 'yellow'

circles = [
    Circle(-1, -2, 0.8, CLASS_BLUE),
    Circle(2, 2, 0.8, CLASS_BLUE),
    Circle(-2.5, 2, 0.8, CLASS_BLUE),
    Circle(-.5, .5, 1.15, CLASS_RED),
]

NUM_POINTS = 1024

for circle in circles:
    circle.points = [generate_point_in_circle(circle) for _ in range(NUM_POINTS)]

if args.show:
    fig, ax = plt.subplots()

    for circle in circles:
        ax.add_patch(patches.Circle((circle.x, circle.y), circle.r, color=class2color(circle.c), fill=False))
        plt.scatter([p.x for p in circle.points], [p.y for p in circle.points], c=class2color(circle.c))

    # Add title and labels
    plt.title('Scatter Plot of 2D Points with Different Colors')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Show color bar
    plt.colorbar(label='Color Scale')

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    # Show the plot
    plt.show()

import torch
from safetensors.torch import save_file

xs  = []
ygt = []
for circle in circles:
    xs.extend([[p.x, p.y] for p in circle.points])
    ygt.extend([circle.c] * len(circle.points))

dtype = torch.float32

xs  = torch.tensor(xs, dtype = dtype)
ygt = torch.tensor(ygt, dtype = dtype)

print(xs.shape)
print(xs)
print(ygt.shape)
print(ygt)

tensors = {
    "x": xs,
    "y": ygt,
}

save_file(tensors, args.out)

