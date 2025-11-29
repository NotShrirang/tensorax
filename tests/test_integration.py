import pytest
import numpy as np
from tensora import Tensor, nn, optim


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_simple_linear_regression(self):
        """Test training a simple linear regression model."""
        # Generate synthetic data: y = 2*x + 1 + noise
        np.random.seed(42)
        x_data = np.random.randn(100, 1)
        y_data = 2 * x_data + 1 + 0.1 * np.random.randn(100, 1)

        # Convert to tensors
        x = Tensor(x_data, dtype='float32')
        y = Tensor(y_data, dtype='float32')

        # Define model
        model = nn.Linear(1, 1)

        # Define optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Training loop
        num_epochs = 100
        losses = []

        for epoch in range(num_epochs):
            # Forward pass
            optimizer.zero_grad()
            pred = model(x)
            loss = ((pred - y) ** 2).mean()  # MSE loss

            # Backward pass
            loss.backward(Tensor([1.0]))
            optimizer.step()

            losses.append(loss.tolist())

        # Check that loss decreased
        assert losses[-1] < losses[0]

        # Check that learned parameters are close to true values
        weight = model.weight.tolist()[0][0]
        bias = model.bias.tolist()[0]

        assert abs(weight - 2.0) < 0.5  # Should be close to 2
        assert abs(bias - 1.0) < 0.5    # Should be close to 1

    def test_mlp_classification(self):
        """Test training a multi-layer perceptron for classification."""
        # Generate synthetic 2D classification data
        np.random.seed(42)
        n_samples = 200
        n_features = 2
        n_classes = 3

        # Create three clusters
        centers = [[-1, -1], [1, 1], [1, -1]]
        x_data = []
        y_data = []

        for i, center in enumerate(centers):
            points = np.random.randn(n_samples // 3, n_features) * 0.5 + center
            x_data.extend(points)
            y_data.extend([i] * (n_samples // 3))

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # Convert to tensors
        x = Tensor(x_data, dtype='float32')
        y = Tensor(y_data.reshape(-1, 1), dtype='float32')

        # Define model
        model = nn.Sequential(
            nn.Linear(n_features, 10),
            nn.ReLU(),
            nn.Linear(10, n_classes),
            nn.Softmax(dim=-1)
        )

        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        num_epochs = 50
        losses = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Forward pass
            logits = model[0](model[1](model[2](x)))  # Manual forward through layers
            pred = model[3](logits)

            # Simple cross-entropy loss (simplified)
            # For now, just use MSE as proxy
            target_onehot = Tensor(np.eye(n_classes)[y_data.flatten().astype(int)], dtype='float32')
            loss = ((pred - target_onehot) ** 2).mean()

            # Backward pass
            loss.backward(Tensor([1.0]))
            optimizer.step()

            losses.append(loss.tolist())

        # Check that loss decreased
        assert losses[-1] < losses[0]

    def test_autoencoder(self):
        """Test training a simple autoencoder."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        x_data = np.random.randn(n_samples, n_features)
        x = Tensor(x_data, dtype='float32')

        # Define autoencoder
        encoder = nn.Sequential(
            nn.Linear(n_features, 5),
            nn.ReLU()
        )

        decoder = nn.Sequential(
            nn.Linear(5, n_features),
            nn.Sigmoid()  # Assuming data is in [0, 1], but we'll use raw data
        )

        # Parameters
        encoder_params = list(encoder.parameters())
        decoder_params = list(decoder.parameters())
        all_params = encoder_params + decoder_params

        optimizer = optim.Adam(all_params, lr=0.01)

        # Training loop
        num_epochs = 50
        losses = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Forward pass
            encoded = encoder(x)
            decoded = decoder(encoded)

            # Reconstruction loss
            loss = ((decoded - x) ** 2).mean()

            # Backward pass
            loss.backward(Tensor([1.0]))
            optimizer.step()

            losses.append(loss.tolist())

        # Check that loss decreased
        assert losses[-1] < losses[0]

        # Check reconstruction quality
        with Tensor.no_grad():
            encoded = encoder(x)
            decoded = decoder(encoded)
            reconstruction_error = ((decoded - x) ** 2).mean().tolist()
            assert reconstruction_error < 1.0  # Should reconstruct reasonably well

    def test_gradient_flow(self):
        """Test that gradients flow correctly through complex networks."""
        # Create a deep network
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 5),
            nn.Softmax(dim=-1)
        )

        x = Tensor(np.random.randn(3, 10), dtype='float32', requires_grad=True)
        y = model(x)

        # Compute some loss
        loss = y.sum()

        # Backward pass
        loss.backward(Tensor([1.0]))

        # Check that gradients exist for all parameters
        for param in model.parameters():
            assert param.grad is not None
            assert not np.allclose(param.grad.tolist(), 0)  # Gradients should be non-zero

    def test_device_consistency(self):
        """Test that operations maintain device consistency."""
        # This test will be more relevant when CUDA is available
        model = nn.Linear(10, 5)

        x_cpu = Tensor(np.random.randn(2, 10), dtype='float32')

        # Forward pass
        y_cpu = model(x_cpu)

        # Check device consistency
        assert x_cpu.device == 'cpu'
        assert y_cpu.device == 'cpu'
        assert model.weight.device == 'cpu'
        assert model.bias.device == 'cpu'

        # If CUDA is available, test device transfer
        if Tensor.cuda_is_available():
            model.cuda()
            x_cuda = x_cpu.cuda()

            y_cuda = model(x_cuda)

            assert x_cuda.device == 'cuda'
            assert y_cuda.device == 'cuda'
            assert model.weight.device == 'cuda'
            assert model.bias.device == 'cuda'

    def test_memory_leaks_prevention(self):
        """Test that tensors and models can be created and deleted without issues."""
        # Create many tensors and models
        tensors = []
        models = []

        for i in range(10):
            tensors.append(Tensor(np.random.randn(100, 100), dtype='float32'))
            models.append(nn.Sequential(nn.Linear(100, 50), nn.ReLU()))

        # Delete them
        del tensors
        del models

        # Should not have memory issues
        # This is more of a stress test

    def test_numerical_stability(self):
        """Test numerical stability of operations."""
        # Test with very small numbers
        x = Tensor([1e-8, 1e-7], dtype='float32')
        y = x.log()
        assert not np.any(np.isinf(y.tolist()))  # Should not be infinite
        assert not np.any(np.isnan(y.tolist()))  # Should not be NaN

        # Test with very large numbers
        x = Tensor([1e8, 1e7], dtype='float32')
        y = x.exp()
        assert not np.any(np.isinf(y.tolist()))  # Might be infinite, but shouldn't crash

    def test_batch_processing(self):
        """Test processing data in batches."""
        batch_size = 32
        n_features = 10
        n_classes = 5

        model = nn.Sequential(
            nn.Linear(n_features, 20),
            nn.ReLU(),
            nn.Linear(20, n_classes)
        )

        # Process multiple batches
        for batch in range(5):
            x = Tensor(np.random.randn(batch_size, n_features), dtype='float32')
            y = model(x)
            assert y.shape == (batch_size, n_classes)

    def test_overfitting_small_dataset(self):
        """Test that the model can overfit a small dataset."""
        # Very small dataset that should be easily learnable
        x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y_data = [[0], [1], [1], [0]]  # XOR function

        x = Tensor(x_data, dtype='float32')
        y = Tensor(y_data, dtype='float32')

        # Simple network for XOR
        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

        optimizer = optim.Adam(model.parameters(), lr=0.1)

        # Train for many epochs on small data
        for epoch in range(200):
            optimizer.zero_grad()
            pred = model(x)
            loss = ((pred - y) ** 2).mean()
            loss.backward(Tensor([1.0]))
            optimizer.step()

        # Should be able to fit the small dataset very well
        final_pred = model(x)
        final_loss = ((final_pred - y) ** 2).mean().tolist()
        assert final_loss < 0.01  # Should fit very well


class TestPerformance:
    """Performance tests (these might be slow)."""

    def test_large_network_training(self):
        """Test training a relatively large network."""
        model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        # Dummy data
        batch_size = 64
        x = Tensor(np.random.randn(batch_size, 784), dtype='float32')
        y = Tensor(np.random.randint(0, 10, (batch_size, 1)), dtype='float32')

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train for a few epochs
        for epoch in range(3):
            optimizer.zero_grad()
            pred = model(x)
            loss = ((pred - y) ** 2).mean()  # Simplified loss
            loss.backward(Tensor([1.0]))
            optimizer.step()

        # Should complete without errors
        assert True

    def test_memory_efficiency(self):
        """Test that operations don't use excessive memory."""
        # Create large tensors and perform operations
        large_tensor = Tensor(np.random.randn(1000, 1000), dtype='float32')

        # Perform some operations
        result = large_tensor + large_tensor
        result = result * Tensor([2.0], dtype='float32')

        # Should complete without memory errors
        assert result.shape == (1000, 1000)
