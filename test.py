import torch
import torch.nn as nn
import torch.optim as optim

class AdaptingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdaptingLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_layer = nn.Linear(input_dim, output_dim)
        self.F = nn.Parameter(torch.full((output_dim,), 0.8, dtype=torch.float64))
        self.tau = nn.Parameter(torch.full((output_dim,), 0.03, dtype=torch.float64))
        self.prev_a = None
        self.prev_y = None

    def forward(self, x):
        out = self.linear_layer(x)
        batch_size = x.size(0)
        if self.prev_a is None or self.prev_a.size(0) != batch_size:
            self.prev_a = torch.zeros(batch_size, self.linear_layer.out_features).to(x.device)
        if self.prev_y is None or self.prev_y.size(0) != batch_size:
            self.prev_y = torch.zeros(batch_size, self.linear_layer.out_features).to(x.device)
        a = self.prev_a + self.tau * (-self.prev_a + self.F * self.prev_y)
        y = out - a
        self.prev_y = y.detach()
        self.prev_a = a.detach()
        return y

    def reset_states(self):
        self.prev_a = None
        self.prev_y = None

class AdaptingMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(AdaptingMLP, self).__init__()
        layers = []
        self.adapting_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            adapting_layer = AdaptingLayer(prev_dim, hidden_dim)
            layers.append(adapting_layer)
            self.adapting_layers.append(adapting_layer)
            prev_dim = hidden_dim
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        x = self.layers(x)
        logits = self.output_layer(x)
        return logits

    def reset_all_states(self):
        for layer in self.adapting_layers:
            layer.reset_states()

# Example usage

def create_sequences(data, window_size):
    sequences = []
    targets = []
    for i in range(len(data) - window_size):
        seq = data[i:i + window_size]
        target = data[i + window_size]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def train_and_evaluate(data, window_size, hidden_layers, output_dim, learning_rate, num_epochs):
    x_sequences, x_targets = create_sequences(data, window_size)
    
    # Convert data to float64
    x_sequences = np.array(x_sequences, dtype=np.float64)
    x_targets = np.array(x_targets, dtype=np.float64)
    
    # Split data
    split_idx = int(0.7 * len(x_sequences))
    x_train = torch.tensor(x_sequences[:split_idx], dtype=torch.float64)
    x_test = torch.tensor(x_sequences[split_idx:], dtype=torch.float64)
    y_train = torch.tensor(x_targets[:split_idx], dtype=torch.float64)
    y_test = torch.tensor(x_targets[split_idx:], dtype=torch.float64)

    # Define model
    model = AdaptingMLP(window_size, hidden_layers, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    model.reset_all_states()
    with torch.no_grad():
        predicted_output = model(x_test)
        test_loss = criterion(predicted_output.squeeze(), y_test)
        print(f'Test Loss: {test_loss.item():.4f}')

    return model

# Example data
import numpy as np
data = np.sin(np.linspace(0, 100, 1000))
window_size = 10
hidden_layers = [10]
output_dim = 1
learning_rate = 0.0001
num_epochs = 10000

model = train_and_evaluate(data, window_size, hidden_layers, output_dim, learning_rate, num_epochs)
