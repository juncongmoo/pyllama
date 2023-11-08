import torch
import torch.nn as nn
from hiq.vis import print_model

# Define a simple neural network with a frozen pretrained layer
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.frozen_layer = nn.Linear(10, 5, bias=False)
        self.frozen_layer.weight.requires_grad = False  # Freeze the weights of this layer
        self.nonfrozen_layer = nn.Linear(5, 1, bias=False)

    def forward(self, x_):
        x_ = self.frozen_layer(x_)
        x_ = self.nonfrozen_layer(x_)
        return x_

# Create a dummy input tensor and a dummy target tensor
x = torch.randn(1, 10)
y = torch.randn(1, 1)

# Create an instance of the network and define the loss function and optimizer
net = MyNet()

print_model(net)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(net.nonfrozen_layer.parameters(), lr=0.1)

# Compute the forward pass and loss
outputs = net(x)
loss = criterion(outputs, y)

# Compute the gradients using automatic differentiation and the chain rule
optimizer.zero_grad()
loss.backward()

print_model(net)

# Print the gradients of the non-frozen weights and the frozen weights
print("Non-frozen weights gradients:")
print(net.nonfrozen_layer.weight.grad)
print(net.nonfrozen_layer.bias)

print("Frozen weights gradients:")
print(net.frozen_layer.weight.grad)
print(net.frozen_layer.bias)
