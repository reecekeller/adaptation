import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from kalman import *
from tqdm import tqdm 

class rnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(rnn, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":

    # Simulation time
    T = 20
    dt = 0.01
    t_space = np.arange(0, T, dt)
    
    # Simulate true dynamics and generate noisy measurements
    var_x = 0.1
    var_z = 0.2
    var_y = 0.1
    

    # Define System Dynamics
    A = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    H = np.array([10, 0.0, 0.5]).reshape((1, 3))
    Q = np.diag([var_x, var_x, var_z])
    R = np.array([[var_y]])
    

    initial_state = np.array([0, 0, 0]).reshape((1, 3))
    initial_covariance = np.diag([0.01, 0.25, 0.01])
    state_vec = dynamics(initial_state, t_space, A, dt, var_x, var_z)
    
    # Get Observations
    data = observations(H, state_vec, var_y)
    
    x = state_vec[:, 0]
    v = state_vec[:, 1]
    z = state_vec[:, 2]
    
    # Run Filter
    state_hat = kalman_filter(A, H, Q, R, initial_state, initial_covariance, data)
    x_hat = state_hat[:, 0]
    v_hat = state_hat[:, 1]
    z_hat = state_hat[:, 2]
    data = np.squeeze(data, 1)

    
    # RNN Parameters
    train_size = int(0.7 * len(data))
    seq_length = 1
    input_size = 1
    output_size = 1
    hidden_size = 32
    
    # Split the data into training and testing sets
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Prepare input and target sequences for training set
    train_input_seq = []
    train_target_seq = []
    for i in range(len(train_data) - seq_length):
        train_input_seq.append(train_data[i:i + seq_length])
        train_target_seq.append(train_data[i + seq_length])
    
    train_input_seq = np.array(train_input_seq).reshape(-1, seq_length, input_size)
    train_target_seq = np.array(train_target_seq).reshape(-1, output_size)
    
    # Prepare input and target sequences for testing set
    test_input_seq = []
    test_target_seq = []
    for i in range(len(test_data) - seq_length):
        test_input_seq.append(test_data[i:i + seq_length])
        test_target_seq.append(test_data[i + seq_length])
    
    entire_dataset_seq = []
    for i in range(len(data) - seq_length):
        entire_dataset_seq.append(data[i:i + seq_length])

    test_input_seq = np.array(test_input_seq).reshape(-1, seq_length, input_size)
    test_target_seq = np.array(test_target_seq).reshape(-1, output_size)
    
    # Convert data to PyTorch tensors
    train_input_seq = torch.tensor(train_input_seq, dtype=torch.float32)
    train_target_seq = torch.tensor(train_target_seq, dtype=torch.float32)
    test_input_seq = torch.tensor(test_input_seq, dtype=torch.float32)
    test_target_seq = torch.tensor(test_target_seq, dtype=torch.float32)
    entire_dataset_tensor = torch.tensor(entire_dataset_seq, dtype=torch.float32)

    # Instantiate the model, loss function, and optimizer
    model = rnn(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    series = []
    
    # Training the model
    num_epochs = 10000
    #for epoch in tqdm(range(num_epochs), desc = f"Training for...{num_epochs}", colour = "magenta"):
    for epoch in range(num_epochs):
        # Forward pass
        output = model(train_input_seq)
        loss = criterion(output, train_target_seq)
    
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # test loss
        predicted_output = model(test_input_seq)
        series.extend(predicted_output)
        test_loss = criterion(predicted_output, test_target_seq)
        
        # Print the loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Testing Loss: {test_loss.item():.4f}')

    
    predicted_outputs = model(entire_dataset_tensor.unsqueeze(2))
    predicted_outputs = predicted_outputs.detach().numpy()

    plt.plot(t_space, x, label='True Position')
    plt.plot(t_space, x_hat, 'g', label = 'Filtered ', linestyle='--')
    plt.plot(t_space[seq_length:], predicted_outputs, label='RNN Predictions', color='red')
    #plt.axvline(x=dt*train_size, color='r', linestyle='--', label='Training Set')

    plt.legend()
 
