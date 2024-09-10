import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Define the true system dynamics
# def dynamics(initial_state, t_space, A, dt, var_x, var_z):
#     N = len(t_space)
#     x = np.zeros((N, 3))
#     #z = np.random.normal(0, var_z, N) # cgeck implementation 
#     x[0] = initial_state
#     #x[0] = initial_state[0][0:2]
#     #z[0] = initial_state[0][-1]

#     for i in range(N-1):    
#         x_noise = np.random.normal(0, var_x)
#         v_noise = np.random.normal(0, var_x)
#         z_noise = np.random.normal(0, var_z)
#         state_noise = np.array([x_noise, v_noise, z_noise])
#         x[i+1] = A @ x[i] * dt + x[i] + state_noise
#     #state_vec = np.column_stack((x, z))
#     return x

def dynamics(initial_state, t_space, A, dt, var_x, var_z):
    N = len(t_space)
    x = np.zeros((N, 3))
    x[0] = initial_state.flatten()  # Ensure initial state is a 1D array
    
    for i in range(N-1):
        # Calculate state update
        x[i+1] = A @ x[i] * dt + x[i]  # Discrete-time update
        
        # Add noise
        x_noise = np.random.normal(0, var_x, 2)  # Noise for state variables x and v
        z_noise = np.random.normal(0, var_z)    # Noise for z
        
        # Apply noise to the state
        x[i+1][:2] += x_noise  # Apply noise to x and v
        x[i+1][2] += z_noise   # Apply noise to z
    
    return x

def observations(H, states, var_y):
#    y = [H @ state + np.random.normal(0, var_y) for state in states] 
    y = [H @ state for state in states] 

    return np.array(y)

def kalman_filter(A, H, Q, R, initial_state, initial_covariance, state_measurements):
    state_size = 3
    # Initialize state estimate and covariance
    state_estimate = initial_state.T
    covariance_estimate = initial_covariance
    
    # Lists to store results
    filtered_states = []

    for measurement in state_measurements:
        # Prediction step
        state_predict = A @ state_estimate
        covariance_predict =  A @ covariance_estimate @ A.T + Q
        
        # Update step
        kalman_gain = (covariance_predict @ H.T) @ np.linalg.inv(H @ (covariance_predict @ H.T) + R)
        state_estimate = state_predict + kalman_gain @ (measurement - H @ state_predict)
        covariance_estimate = (np.eye(state_size)-kalman_gain @ H) @ covariance_predict

        # Save filtered result
        filtered_states.append(state_estimate)

    return np.array(filtered_states)

if __name__ == "__main__":

    # Simulation time
    T = 10
    dt = 0.1
    t_space = np.arange(0, T, dt)
    
    # Noise Parameters
    var_x = 0.1
    var_z = 0.2
    var_y = 0.1
    
    # Define and Simulate System Dynamics
    A = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0.01]])
    H = np.array([1.0, 0.0, 0.6]).reshape((1, 3))
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
    
    # Plot results
    plt.plot(t_space, x, label='True Position')
    plt.plot(t_space, z, '-r', label='Nuisance Variable')
    plt.plot(t_space, y, 'm', label='Observations')
    plt.plot(t_space, x_hat, 'g', label = 'Filtered ', linestyle='--')
    
    plt.ylim([-3, 3])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('1D Oscillatory System')
    plt.legend()
    plt.show()
