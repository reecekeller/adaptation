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
        
        self.W = nn.Parameter(torch.ones(output_dim, input_dim, dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(output_dim, dtype=torch.float64))
        self.F = nn.Parameter(torch.full((output_dim,), 0.8, dtype=torch.float64))  # Learnable F parameter
        self.tau = nn.Parameter(torch.full((output_dim,), 0.03, dtype=torch.float64))  # Learnable tau parameter
        
        ## Remove init for comparing against exp filter
        nn.init.xavier_normal_(self.W)
        nn.init.constant_(self.b, 0)
        
        self.register_buffer('prev_y', None)
        self.register_buffer('prev_a', None)
        
        self.prev_y = torch.zeros(output_dim, dtype=torch.float64)
        self.prev_a = torch.zeros(output_dim, dtype=torch.float64)
        
        
    def forward(self, x):
        x = torch.matmul(x, self.W.t()) + self.b
        ## Clamping tau to ensure stability
        # self.tau.data = torch.clamp(self.tau.data, min=1e-4, max=1.0)
        a = self.prev_a + self.tau * (-self.prev_a + self.F * self.prev_y)
        y = x - a
        self.prev_y = y.detach()
        self.prev_a = a.detach()
        return y

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
        logits = self.output_layer(x.to(torch.float64))
        return logits
    
    def get_all_tau(self):
        return [layer.get_tau() for layer in self.adapting_layers]
    def get_all_F(self):
        return [layer.get_F() for layer in self.adapting_layers]
    def get_all_W(self):
        return[layer.get_W() for layer in self.adapting_layers] + [self.output_layer.weight.detach().numpy()]
    def get_all_b(self):
        return[layer.get_b() for layer in self.adapting_layers] + [self.output_layer.bias.detach().numpy()]
        

