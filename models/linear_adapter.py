# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:03:37 2024

@author: reece
"""

import torch
import torch.nn as nn
torch.set_default_dtype(torch. float64)

class AdaptingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdaptingLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear_layer = nn.Linear(input_dim, output_dim)
        self.F = nn.Parameter(torch.full((output_dim,), 0.7, dtype=torch.float64))  # Learnable F parameter
        self.tau = nn.Parameter(torch.full((output_dim,), 0.96, dtype=torch.float64))  # Learnable tau parameter
    
        self.register_buffer('prev_y', None)
        self.register_buffer('prev_a', None)

        self.prev_a = None
        self.prev_y = None
        
    def forward(self, x):
        out = self.linear_layer(x)
        ## Clamping tau to ensure stability
        #self.tau.data = torch.clamp(self.tau.data, min=0.01, max=0.05)
        #self.F.data = torch.clamp(self.F.data, min=1e-1, max=1.0)

        batch_size = x.size(0)

        # Initialize state a and y with zeros on the first forward pass
        if self.prev_a is None:
            self.prev_a = torch.zeros(batch_size, self.linear_layer.out_features).to(x.device)
        if self.prev_y is None:
            self.prev_y = torch.zeros(batch_size, self.linear_layer.out_features).to(x.device)
    
       # a = self.prev_a + self.tau * (-self.prev_a + self.F * self.prev_y)
        #y = out - a
        a =self.tau * self.prev_a + (1-self.tau)*self.prev_y
        y = out - self.F*a
       
        self.prev_y = y.detach()
        self.prev_a = a.detach()

        return y
    def reset_states(self):
        """Reset prev_a and prev_y to None to handle different batch sizes."""
        self.prev_a = None
        self.prev_y = None
    def get_tau(self):
        return self.tau.detach().numpy()

    def get_F(self):
        return self.F.detach().numpy()
    def get_W(self):
        return self.W.detach().numpy()
    def get_b(self):
        return self.b.detach().numpy()

class AdaptingMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(AdaptingMLP, self).__init__()
        layers = []
        self.adapting_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            adapting_layer = AdaptingLayer(prev_dim, hidden_dim)
            layers.append(adapting_layer)
            layers.append(nn.ReLU())
            self.adapting_layers.append(adapting_layer)
            prev_dim = hidden_dim
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        ## initialization to compare against exp filter.
        # with torch.no_grad():
        #     self.output_layer.weight.copy_(torch.ones(prev_dim, output_dim))
        #     self.output_layer.bias.copy_(torch.tensor(0))
    def forward(self, x):
        x = self.layers(x)
        x = self.output_layer(x.to(torch.float64))
        return x
    
    def reset_hidden_states(self):
        """Reset the states of all adapting layers in the network."""
        for layer in self.adapting_layers:
            layer.reset_states()

    def get_all_tau(self):
        return [layer.get_tau() for layer in self.adapting_layers]
    def get_all_F(self):
        return [layer.get_F() for layer in self.adapting_layers]
    def get_all_W(self):
        return[layer.get_W() for layer in self.adapting_layers] + [self.output_layer.weight.detach().numpy()]
    def get_all_b(self):
        return[layer.get_b() for layer in self.adapting_layers] + [self.output_layer.bias.detach().numpy()]

class AdaptingLayergpt(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdaptingLayergpt, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear_layer = nn.Linear(input_dim, output_dim)
        self.F = nn.Parameter(torch.full((output_dim,), 0.7, dtype=torch.float64))  # Learnable F parameter
        self.tau = nn.Parameter(torch.full((output_dim,), 0.96, dtype=torch.float64))  # Learnable tau parameter
    
        self.register_buffer('prev_y', None)
        self.register_buffer('prev_a', None)

        self.prev_a = None
        self.prev_y = None
        
    def forward(self, x, prev_hidden_state=None):
        # x is the current observation
        # prev_hidden_state can be passed externally, if not use the internally stored one
        out = self.linear_layer(x)
        
        batch_size = x.size(0)

        # Initialize state a and y with zeros on the first forward pass
        if self.prev_a is None:
            self.prev_a = torch.zeros(batch_size, self.linear_layer.out_features).to(x.device)
        if self.prev_y is None:
            self.prev_y = torch.zeros(batch_size, self.linear_layer.out_features).to(x.device)

        # Use passed previous hidden state if provided, otherwise use internally stored one
        prev_a = prev_hidden_state if prev_hidden_state is not None else self.prev_a

        # Update a_t and y_t
        a = self.tau * prev_a + (1 - self.tau) * self.prev_y
        y = out - self.F * a
        
        # Store current hidden states
        self.prev_y = y.detach()
        self.prev_a = a.detach()

        return y, a  # Return both the output and hidden state for next layer

    def reset_states(self):
        """Reset prev_a and prev_y to None to handle different batch sizes."""
        self.prev_a = None
        self.prev_y = None
        
class AdaptingMLPgpt(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(AdaptingMLPgpt, self).__init__()
        layers = []
        self.adapting_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            adapting_layer = AdaptingLayer(prev_dim, hidden_dim)
            layers.append(adapting_layer)
            layers.append(nn.ReLU())
            self.adapting_layers.append(adapting_layer)
            prev_dim = hidden_dim
        self.layers = nn.Sequential(*layers)
        
        # Now, the output dimension should be 2 to predict both states
        self.output_layer = nn.Linear(prev_dim, 2)  # Predict both states

    def forward(self, x, prev_hidden_states=None):
        if prev_hidden_states is None:
            prev_hidden_states = [None] * len(self.adapting_layers)
        
        # Pass through adapting layers with hidden state integration
        hidden_states = []
        for i, layer in enumerate(self.adapting_layers):
            x, hidden_state = layer(x, prev_hidden_states[i])
            hidden_states.append(hidden_state)

        # Final output layer for predicting both states
        output = self.output_layer(x.to(torch.float64))

        return output, hidden_states

    def reset_hidden_states(self):
        """Reset the states of all adapting layers in the network."""
        for layer in self.adapting_layers:
            layer.reset_states()     

class linearNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(linearNN, self).__init__()
        layers=[]
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            linear_layer = nn.Linear(prev_dim, hidden_dim)
            layers.append(linear_layer)
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
    def forward(self, x):
        x = self.layers(x)
        x = self.output_layer(x.to(torch.float64))
        return x
