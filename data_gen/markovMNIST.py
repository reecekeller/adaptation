# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:23:47 2024

@author: reece
"""


import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from models.adapter_full import *
import matplotlib.pyplot as plt

class MarkovMNISTDataset(Dataset):
    def __init__(self, mnist_dataset, transition_probs, seq_length=10):
        self.mnist_dataset = mnist_dataset
        self.transition_probs = transition_probs
        self.seq_length = seq_length
        self.digit_indices = {i: [] for i in range(10)}
        for idx, (img, label) in enumerate(self.mnist_dataset):
            self.digit_indices[label].append(idx)

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        start_digit = np.random.choice(10)  # Start from a random digit
        trajectory = [start_digit]
        for _ in range(self.seq_length - 1):
            next_digit = np.random.choice(10, p=self.transition_probs[trajectory[-1]])
            trajectory.append(next_digit)
        
        images = [self.mnist_dataset[np.random.choice(self.digit_indices[digit])][0] for digit in trajectory]
        images = torch.stack(images)
        return images, torch.tensor(trajectory)

# Testing function
def test_model(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation
        for images, trajectories in test_loader:
            # Flatten the images for each sequence
            batch_size, seq_length, _, _, _ = images.shape
            images = images.view(batch_size * seq_length, 28 * 28).to(device)
            trajectories = trajectories.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Reshape outputs and targets for the loss computation
            outputs = outputs.view(batch_size, seq_length, -1)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = trajectories.view(-1)
            
            # Compute loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Get the predicted class with the highest score
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    # Calculate average loss and accuracy
    average_loss = test_loss / len(test_loader)
    accuracy = correct / total
    print(f'Average Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    return average_loss, accuracy

input_dim = 28 * 28  # MNIST images are 28x28
hidden_layers = [32, 64, 128, 64, 32]  # Example hidden layer sizes
output_dim = 10  # MNIST has 10 classes (digits 0-9)
batch_size = 20
learning_rate = 0.001  
num_epochs = 5

# Initialize model, loss function, and optimizer
model = AdaptingMLP(input_dim, hidden_layers, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Create the Markov MNIST dataset
# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Define the transition probabilities for the Markov chain
# For simplicity, let's create a transition matrix where the probability to transition to the next digit is higher
transition_probs = np.full((10, 10), 0.1)
for i in range(10):
    transition_probs[i, (i + 1) % 10] = 0.8  # Increase probability to transition to the next digit
    transition_probs[i] /= transition_probs[i].sum()  # Normalize to sum to 1

mmd_train = MarkovMNISTDataset(mnist_train, transition_probs, seq_length=10)
mmd_test = MarkovMNISTDataset(mnist_test, transition_probs, seq_length=10)

train_loader = DataLoader(mmd_train, batch_size, shuffle=True)
test_loader = DataLoader(mmd_test, batch_size, shuffle=False)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, trajectories) in enumerate(train_loader):
        # Flatten the images for each sequence
        batch_size, seq_length, _, _, _ = images.shape
        images = images.view(batch_size * seq_length, 28 * 28).to(device)
        trajectories = trajectories.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Reshape outputs and targets for the loss computation
        outputs = outputs.view(batch_size, seq_length, -1)
        outputs = outputs.view(-1, outputs.size(-1))
        targets = trajectories.view(-1)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate running loss
        running_loss += loss.item()
        
        if batch_idx % 100 == 99:  # Print every 100 batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print('Finished Training')

# Evaluate the model on the test dataset
test_loss, test_accuracy = test_model(model, test_loader, criterion)

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
plt.title('Timescale Distribution Across Layers')
plt.xticks(np.arange(1, len(all_F) + 1), labels=np.arange(1, len(all_tau) + 1))  # Set x-axis ticks and labels
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()
