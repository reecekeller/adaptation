# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:39:16 2024

@author: reece
"""

import numpy as np
import matplotlib.pyplot as plt
from models.adapter_full import *

# Define parameters
time = np.linspace(0, 15, 1000)  # Time from 0 to 15 with 1000 points
dt = time[1] - time[0]  # Time step
step_time = 5  # Time when the step stimulus occurs
stimulus_duration = 3  # Duration of the step stimulus
step_value = 1.0  # Value of the step stimulus
s = np.where((time >= step_time) & (time <= step_time + stimulus_duration), step_value, 0)  # Step stimulus
F=0.8 # adjusts adaptation threshold (asymptote)
tau=0.03 # adaptation timescale 
s_t = torch.tensor(s, dtype=torch.float64).unsqueeze(1)

gamma=1
c = 1
input_dim = 1  
hidden_layers = [1] 
output_dim = 1  

y = np.zeros_like(s)
a = np.zeros_like(s)
m = np.zeros_like(s)
model = AdaptingMLP(input_dim, hidden_layers, output_dim)

# Simulate activity of the neurons
for t in range(1, len(time)):
    a[t] = a[t-1] + tau * (-a[t-1] + F*y[t-1])
    y[t] = np.maximum(gamma*s[t]-c*a[t], 0)
    m[t] = model(s_t[t])
    

# Initialize model, loss function, and optimizer


all_tau = model.get_all_tau()
all_F = model.get_all_F()
all_W = model.get_all_W()
all_b = model.get_all_b()

print(all_tau, all_F, all_W, all_b)
    
    
# Plot stimulus and neuron activity
plt.plot(time, s, label='Stimulus')
plt.plot(time, y, label='Neuron Activity')
plt.plot(time, a, label='Adaptation Level')
plt.plot(time, m, label='Network', ls='--')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('ReLU Neuron Response Unit Step Stimulus')
plt.legend()
plt.grid(True)
plt.show()
