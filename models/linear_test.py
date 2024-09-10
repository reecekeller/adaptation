from linear_adapter import *
from kalman import *
import torch.optim as optim
import numpy as np

# Simulation time
T = 10
dt = 0.01
t_space = np.arange(0, T, dt)

# Noise Parameters
var_x = 0.1
var_z = 0.2
var_y = 0.1

# Define and Simulate System Dynamics
A = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0.01]])
H = np.array([3.0, 0.0, 0.6]).reshape((1, 3))
Q = np.diag([var_x, var_x, var_z])
R = np.array([[var_y]])

initial_state = np.array([0, 0, 0]).reshape((1, 3))
initial_covariance = np.diag([0.01, 0.25, 0.01])
state_vec = dynamics(initial_state, t_space, A, dt, var_x, var_z)

# Get State Variables
x = state_vec[:, 0]
v = state_vec[:, 1]
z = state_vec[:, 2]

# Get Observations
y = observations(H, state_vec, var_y)

# Run Filter
state_hat = kalman_filter(A, H, Q, R, initial_state, initial_covariance, y)
x_hat = state_hat[:, 0]
v_hat = state_hat[:, 1]
z_hat = state_hat[:, 2]

# Data Preparation
# Create input-output pairs for time series prediction
window_size = 10  # Define the size of the window for time series prediction
def create_sequences(data, window_size):
    sequences = []
    targets = []
    for i in range(len(data) - window_size):
        seq = data[i:i + window_size]
        target = data[i + window_size]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Prepare sequences for training
x_sequences, x_targets = create_sequences(x, window_size)
obs_sequences, obs_targets = create_sequences(y, window_size)

# Train/Test Split
split_idx = int(0.7 * len(x_sequences))
x_train = torch.tensor(x_targets[:split_idx])
x_test = torch.tensor(x_targets[split_idx:])
obs_train = torch.tensor(x_sequences[:split_idx])
obs_test = torch.tensor(x_sequences[split_idx:])

# Define Adapting Network
input_dim = window_size
hidden_layers = [10]
output_dim = 1

learning_rate = 0.01
num_epochs = 1000

linear_model = AdaptingMLP(input_dim, hidden_layers, output_dim)
ff_model = linearNN(input_dim, hidden_layers, output_dim)

criterion = nn.MSELoss()
optimizer1 = optim.Adam(linear_model.parameters(), lr=learning_rate)
optimizer2 = optim.Adam(ff_model.parameters(), lr=learning_rate)
.
#Training loop
linear_model.train()
for epoch in range(num_epochs):
### Adaptive Network
    output1 = linear_model(obs_train)
    loss1 = criterion(output1.squeeze(), x_train)
    
    optimizer1.zero_grad()
    loss1.backward()
    optimizer1.step()

### Linear FF Network
    output2 = ff_model(obs_train)
    loss2 = criterion(output2.squeeze(), x_train)
    
    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()
    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Adp. Training Loss: {loss1.item():.4f}, FF Training Loss: {loss2.item():4f}')
linear_model.eval()
linear_model.reset_hidden_states()
predicted_output = linear_model(obs_test)
test_loss = criterion(predicted_output, x_test)
print(f'Test Loss: {test_loss.item():.4f}')

linear_model.reset_hidden_states()

x_seq = torch.tensor(x_sequences)
model_rollout = linear_model(x_seq)

# Plot results
plt.plot(t_space, x, label='True Position')
#plt.plot(t_space, z, '-r', label='Nuisance Variable')
plt.plot(t_space, y, 'm', label='Observations')
plt.plot(t_space, x_hat, 'g', label = 'Kalman Estimate ', linestyle='--')
plt.plot(t_space[window_size:], model_rollout.detach().numpy(), 'r', label = 'Network Estimate', linestyle='--')
#plt.ylim([-3, 3])
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('1D Oscillatory System')
plt.legend()
plt.show()

all_tau = linear_model.get_all_tau()
all_F = linear_model.get_all_F()

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

print(all_tau, all_F)