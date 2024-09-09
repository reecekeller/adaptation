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
H = np.array([3.0, 0.0, 1.0]).reshape((1, 3))
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

# Train/Test Split
split_idx = int(0.7*len(x_hat))
x_train = torch.tensor(x[:split_idx]).unsqueeze(1)
x_test = torch.tensor(x[split_idx:]).unsqueeze(1)
obs_train = torch.tensor(y[:split_idx])
obs_test = torch.tensor(y[split_idx:])

#print(x_train.unsqueeze(1).size())
# Define Adapting Network
input_dim = 1
hidden_layers = [10]
output_dim = 1

learning_rate = 0.0001
num_epochs = 10000

linear_model = AdaptingMLP(input_dim, hidden_layers, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(linear_model.parameters(), lr=learning_rate)

#Training loop
linear_model.train()
for epoch in range(num_epochs):
    output = linear_model(obs_train)
    loss = criterion(output, x_train)
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}')
linear_model.eval()
linear_model.reset_hidden_states()
predicted_output = linear_model(obs_test)
test_loss = criterion(predicted_output, x_test)
print(f'Test Loss: {test_loss.item():.4f}')

x = torch.tensor(x).unsqueeze(1)
linear_model.reset_hidden_states()
model_rollout = linear_model(x)

# Plot results
plt.plot(t_space, x.detach().numpy(), label='True Position')
#plt.plot(t_space, z, '-r', label='Nuisance Variable')
plt.plot(t_space, y, 'm', label='Observations')
plt.plot(t_space, x_hat, 'g', label = 'Kalman Estimate ', linestyle='--')
plt.plot(t_space, model_rollout.detach().numpy(), 'r', label = 'Network Estimate', linestyle='--')
#plt.ylim([-3, 3])
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('1D Oscillatory System')
plt.legend()
plt.show()