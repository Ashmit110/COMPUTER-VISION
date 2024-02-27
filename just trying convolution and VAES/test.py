import torch
import torch.nn as nn

# Example Flatten layer
flatten_layer = nn.Flatten(start_dim=1)

# Example input tensor (batch_size, channels, height, width)
input_tensor = torch.randn(32, 3, 128, 128)

# Forward pass through the Flatten layer
flattened_tensor = flatten_layer(input_tensor)
print(flattened_tensor.shape)