# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:03:37 2024

@author: reece
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class AdaptingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, tau, F):
        super(AdaptingLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau = tau
        self.F = F
        
        self.W = nn.Parameter(torch.zeros(output_dim, input_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))   
        nn.init.xavier_normal_(self.W)
        self.register_buffer('prev_y', None)
        self.register_buffer('prev_a', None)

    def forward(self, x):
        x = torch.matmul(x, self.W.t()) + self.b
        if self.prev_y is None or self.prev_y.size(0) != x.size(0):
            self.prev_y = torch.zeros(x.size(0), self.output_dim, device=x.device)
            self.prev_a = torch.zeros(x.size(0), self.output_dim, device=x.device)
        a = self.prev_a + self.tau * (self.prev_a - self.F * self.prev_y)
        y = x - a
        self.prev_y = y.detach()
        self.prev_a = a.detach()
        return y

class AdaptingMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, tau, F):
        super(AdaptingMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(AdaptingLayer(prev_dim, hidden_dim, tau, F))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
    def forward(self, x):
        x = self.layers(x)
        logits = self.output_layer(x)
        return logits

# Hyperparameters
F = 0.0  # adjusts adaptation threshold (asymptote)
tau = 0.0  # adaptation timescale 
input_dim = 28 * 28  # MNIST images are 28x28
hidden_layers = [128, 64]  # Example hidden layer sizes
output_dim = 10  # MNIST has 10 classes (digits 0-9)
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = AdaptingMLP(input_dim, hidden_layers, output_dim, tau, F)
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
