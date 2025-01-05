#!/bin/python3
# vim: set textwidth=120 shiftwidth=4 softtabstop=4 expandtab:

import argparse

parser = argparse.ArgumentParser(
    description = "Generates groups of 2d points for the classification problem. Generates both training and test data")

parser.add_argument('-o', '--out', default = '', type = str, help = "Output file")
parser.add_argument('-c', '--compute', default = False, action = 'store_true',
                    help = "Compute and show results locally, using pytorch")

args = parser.parse_args()

if args.out == '' and args.compute == False:
    print('Either --compute or --out is required (or both).')
    parser.print_help()
    from sys import exit
    exit(0)

import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

if args.compute:

    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt

    # Define the neural network model
    class ClassifyNN(nn.Module):
        def __init__(self):
            super(ClassifyNN, self).__init__()
            self.fc1 = nn.Linear(2, 10)
            self.fc2 = nn.Linear(10, 2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Initialize the model, loss function, and optimizer
    model = ClassifyNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 1000
    print(f'Epoch [0/{num_epochs}]', end = '', flush = True)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(X_train_tensor)  # Forward pass
        loss = criterion(outputs, y_train_tensor)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        if (epoch + 1) % 10 == 0:
            print(f'\033[2K\rEpoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}', end = '', flush = True)
    print('')

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f'Accuracy: {accuracy * 100:.2f}%')

    # Plotting decision boundary
    def plot_decision_boundary(model, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001),
                             np.arange(y_min, y_max, 0.001))
        Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
        _, Z = torch.max(Z.data, 1)
        Z = Z.numpy().reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
        plt.title('Decision Boundary')
        plt.show()

    plot_decision_boundary(model, X_test, y_test)

if args.out:

    from safetensors.torch import save_file

    tensors = {
        "xtrain": X_train_tensor.to(torch.float32),
        "ytrain": y_train_tensor.to(torch.float32),
        "xtest": X_test_tensor.to(torch.float32),
        "ytest": y_test_tensor.to(torch.float32),
    }

    save_file(tensors, args.out)

    print(X_train_tensor)
    print(X_train_tensor.shape)

    print(y_train_tensor)
    print(y_train_tensor.shape)

    print(X_test_tensor)
    print(X_test_tensor.shape)

    print(y_test_tensor)
    print(y_test_tensor.shape)

