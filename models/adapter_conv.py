import torch
import torch.nn as nn
import torch.nn.functional as F
class adaptiveLinearLayer(nn.Module):
    def __init__(self, alpha, beta, input_dim, output_dim):
        super(adaptiveLinearLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.beta = beta

        self.linear = nn.Linear(input_dim, output_dim)

        self.register_buffer('prev_activation', None)
        self.register_buffer('prev_adapation', None)

        self.prev_adapation = None
        self.prev_activation = None
        
    def forward(self, x):

        # Compute the current activation
        curr_activation = self.linear(x)
        adaptation_integrator = torch.zeros_like(curr_activation)
        # If there is a previous activation, subtract it from the current one
        if self.prev_activation is not None:
            adaptation_integrator = self.prev_adaptation + self.beta * (-self.prev_adaptation + self.alpha * self.prev_activation)
            curr_activation = curr_activation - adaptation_integrator
        # Store the current activation for the next forward pass
        self.prev_activation = curr_activation.clone().detach()
        self.prev_adaptation = adaptation_integrator.clone().detach()
        return curr_activation


    def get_tau(self):
        return self.tau.detach().numpy()
    def get_F(self):
        return self.F.detach().numpy()
    def get_W(self):
        return self.W.detach().numpy()
    def get_b(self):
        return self.b.detach().numpy()

class adaptiveConvBlock(nn.Module):
    def __init__(self, alpha, beta, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(adaptiveConvBlock, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
       
        self.register_buffer('prev_activation', None)
        self.register_buffer('prev_adapation', None)

        self.prev_adaptation = None
        self.prev_activation = None

    def forward(self, x):
        # Compute the current activation
        curr_activation = self.conv(x)
        adaptation_integrator = torch.zeros_like(curr_activation)
        # If there is a previous activation, subtract it from the current one
        if self.prev_activation is not None:
            adaptation_integrator = self.prev_adaptation + self.beta * (-self.prev_adaptation + self.alpha * self.prev_activation)
            curr_activation = curr_activation - adaptation_integrator
        
        # Store the current activation for the next forward pass
        self.prev_activation = curr_activation.clone().detach()
        self.prev_adaptation = adaptation_integrator.clone().detach()
        return curr_activation

class AdapationNet(nn.Module):
    def __init__(self, alpha = 0.8, beta = 0.03, num_classes=10):  # Adjust num_classes as per the task
        super(AdapationNet, self).__init__()
        
        # First convolutional layer: 32 filters of size 5x5x1, stride 1
        self.conv1 = adaptiveConvBlock(alpha, beta, in_channels=1, out_channels=32, kernel_size=5, stride=1)
        
        # Second convolutional layer: 32 filters of size 5x5x32, stride 1
        self.conv2 = adaptiveConvBlock(alpha, beta, in_channels=32, out_channels=32, kernel_size=5, stride=1)
        
        # Third convolutional layer: 32 filters of size 3x3x32, stride 1
        self.conv3 = adaptiveConvBlock(alpha, beta, in_channels=32, out_channels=32, kernel_size=3, stride=1)
        
        # Fully connected layer with 1024 units and 50% dropout
        self.fc1 = adaptiveLinearLayer(alpha, beta, 32 * 3 * 3, 1024)  # Adjust the input size (after flattening)
        self.dropout = nn.Dropout(p=0.5)
        
        # Final output layer (fully connected decoder)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        # Input shape is assumed to be [batch_size, 1, 28, 28] for grayscale images
        # First conv layer followed by ReLU and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Second conv layer followed by ReLU and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Third conv layer followed by ReLU
        x = F.relu(self.conv3(x))
        
        # Flatten the output of the conv layers before the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten into [batch_size, num_features]
        
        # First fully connected layer followed by ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout during training
        
        # Final output layer (logits)
        x = self.fc2(x)
        
        return x

# Instantiate the model
model = AdapationNet(alpha = 0.8, beta = 0.03, num_classes=10)  # For example, 10 classes for classification

# Print the model architecture
print(model)

# Example usage
#input_data = torch.randn(1, 1, 28, 28)  # Batch of 8 grayscale images (1 channel, 28x28 pixels)
#output = model(input_data)
#print(output.shape)  # Should be [8, num_classes]
