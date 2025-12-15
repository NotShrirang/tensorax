"""
Example of building a simple neural network with Tensora.
"""

import numpy as np
from tensorax import Tensor, nn, optim, functional as F

class SimpleNet(nn.Module):
    """Simple feedforward neural network."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    print("=== Simple Neural Network Example ===\n")
    
    # Create model
    model = SimpleNet(input_size=10, hidden_size=20, output_size=5)
    print(f"Model created with {sum(1 for _ in model.parameters())} parameter tensors\n")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Generate random data
    batch_size = 32
    X = Tensor(np.random.randn(batch_size, 10), dtype='float32')
    y_true = Tensor(np.random.randn(batch_size, 5), dtype='float32')
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model(X)
        
        # Compute loss
        loss = F.mse_loss(y_pred, y_true)
        
        # Backward pass (gradient computation)
        optimizer.zero_grad()
        # loss.backward()  # Note: Full autograd will be implemented
        
        # Update parameters
        # optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.numpy():.4f}")
    
    print("\nTraining completed!")
    
    # Evaluation mode
    model.eval()
    test_input = Tensor(np.random.randn(1, 10), dtype='float32')
    test_output = model(test_input)
    print(f"\nTest output shape: {test_output.shape}")

if __name__ == "__main__":
    main()
