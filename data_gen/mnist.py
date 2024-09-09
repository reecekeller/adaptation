# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:26:56 2024

@author: reece
"""
from models.adapter_full import *
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
input_dim = 28 * 28  # MNIST images are 28x28
hidden_layers = [32, 64, 128, 64, 32]  # Example hidden layer sizes
output_dim = 10  # MNIST has 10 classes (digits 0-9)
batch_size = 60
learning_rate = 0.001  # Reduced learning rate
num_epochs = 5

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Initialize model, loss function, and optimizer
model = AdaptingMLP(input_dim, hidden_layers, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)  # Flatten the image
        # Forward pass
        logits = model(data)
        loss = criterion(logits, target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)  # Flatten the image
        logits = model(data)
        probabilities = torch.softmax(logits, dim=1)
        _, predicted = torch.max(probabilities, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')

# Retrieve tau and F parameters after training
all_tau = model.get_all_tau()
all_F = model.get_all_F()

all_tau = model.get_all_tau()
all_F = model.get_all_F()

data_for_boxplot = [list(v) for v in all_tau]

# Plotting
plt.figure(figsize=(10, 6))

plt.boxplot(data_for_boxplot, positions=np.arange(1, len(all_tau) + 1))

# Customize the plot
plt.xlabel('Layer')
plt.ylabel('Tau')
plt.title('Timescale Distribution Across Layers')
plt.xticks(np.arange(1, len(all_tau) + 1), labels=np.arange(1, len(all_tau) + 1))  # Set x-axis ticks and labels
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()

data_for_boxplot = [list(v) for v in all_F]

# Plotting
plt.figure(figsize=(10, 6))

plt.boxplot(data_for_boxplot, positions=np.arange(1, len(all_F) + 1))

# Customize the plot
plt.xlabel('Layer')
plt.ylabel('Tau')
plt.title('Adaptation Threshold Distribution Across Layers')
plt.xticks(np.arange(1, len(all_F) + 1), labels=np.arange(1, len(all_tau) + 1))  # Set x-axis ticks and labels
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()
