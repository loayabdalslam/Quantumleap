import numpy as np
from quantum_leap.tensor import Tensor
from quantum_leap.nn import Module, Linear, ReLU
from quantum_leap.optim import SGD

# 1. Define a simple neural network model
class SimpleNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(10, 50)
        self.relu1 = ReLU()
        self.fc2 = Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# 2. Mean Squared Error Loss function
def mse_loss(predictions, targets):
    # This now returns a Tensor, fully connected to the graph
    diff = predictions + (targets * -1)
    return (diff * diff).mean()

# 3. Create some sample data
X_train = Tensor(np.random.rand(100, 10))
y_train = Tensor(np.random.rand(100, 1))

# 4. Instantiate the model and optimizer
model = SimpleNet()
optimizer = SGD(model.parameters(), lr=0.01)

# 5. Training loop
print("--- Starting Training ---")
for epoch in range(101):
    # Zero the gradients
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(X_train)
    
    # Compute loss
    loss = mse_loss(predictions, y_train)
    
    # Backward pass to compute gradients
    loss.backward()

    # Update weights
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")

print("--- Training Finished ---")
