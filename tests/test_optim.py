import pytest
import numpy as np
from tensorax import Tensor, optim, nn


class TestSGDOptimizer:
    """Test Stochastic Gradient Descent optimizer."""

    def test_sgd_initialization(self):
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        assert optimizer.lr == 0.01
        assert optimizer.momentum == 0.0
        assert len(optimizer.params) == 2  # weight and bias

    def test_sgd_with_momentum(self):
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        assert optimizer.momentum == 0.9

    def test_sgd_step(self):
        # Simple parameter update test
        param = Tensor([2.0], requires_grad=True)
        param.grad = Tensor([1.0])  # Gradient pointing to decrease param

        optimizer = optim.SGD([param], lr=0.1)
        optimizer.step()

        # Parameter should decrease: 2.0 - 0.1 * 1.0 = 1.9
        assert abs(param.tolist()[0] - 1.9) < 1e-5

    def test_sgd_step_with_momentum(self):
        param = Tensor([2.0], requires_grad=True)
        param.grad = Tensor([1.0])

        optimizer = optim.SGD([param], lr=0.1, momentum=0.5)
        optimizer.step()

        # First step with momentum: param = 2.0 - 0.1 * 1.0 = 1.9
        assert abs(param.tolist()[0] - 1.9) < 1e-5

        # Set new gradient
        param.grad = Tensor([0.5])
        optimizer.step()

        # Second step: velocity = 0.5 * velocity + 0.1 * 0.5 = 0.5 * 0.1 + 0.05 = 0.1
        # param = 1.9 - 0.1 = 1.8
        assert abs(param.tolist()[0] - 1.8) < 1e-5

    def test_sgd_zero_grad(self):
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Set some gradients
        for param in model.parameters():
            param.grad = Tensor.ones(param.shape)

        optimizer.zero_grad()

        for param in model.parameters():
            assert param.grad is None

    def test_sgd_multiple_steps(self):
        param = Tensor([10.0], requires_grad=True)

        optimizer = optim.SGD([param], lr=0.1)

        for i in range(5):
            param.grad = Tensor([2.0])  # Constant gradient
            optimizer.step()

        # After 5 steps: 10.0 - 5 * 0.1 * 2.0 = 10.0 - 1.0 = 9.0
        assert abs(param.tolist()[0] - 9.0) < 1e-5

    def test_sgd_no_grad(self):
        param = Tensor([2.0], requires_grad=True)
        # No gradient set

        optimizer = optim.SGD([param], lr=0.1)
        optimizer.step()

        # Parameter should remain unchanged
        assert abs(param.tolist()[0] - 2.0) < 1e-5


class TestAdamOptimizer:
    """Test Adam optimizer."""

    def test_adam_initialization(self):
        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        assert optimizer.lr == 0.001
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.999
        assert optimizer.eps == 1e-8
        assert optimizer.t == 0

    def test_adam_step(self):
        param = Tensor([2.0], requires_grad=True)
        param.grad = Tensor([1.0])

        optimizer = optim.Adam([param], lr=0.1)
        optimizer.step()

        # Adam first step: m = beta1 * m + (1-beta1) * grad = 0.9*0 + 0.1*1 = 0.1
        # v = beta2 * v + (1-beta2) * grad^2 = 0.999*0 + 0.001*1 = 0.001
        # m_hat = m / (1-beta1^t) = 0.1 / (1-0.9) = 0.1 / 0.1 = 1.0
        # v_hat = v / (1-beta2^t) = 0.001 / (1-0.999) = 0.001 / 0.001 = 1.0
        # update = lr * m_hat / sqrt(v_hat + eps) = 0.1 * 1.0 / sqrt(1.0 + 1e-8) ≈ 0.1
        # param = 2.0 - 0.1 ≈ 1.9
        assert abs(param.tolist()[0] - 1.9) < 0.01  # Approximate due to bias correction

    def test_adam_multiple_steps(self):
        param = Tensor([5.0], requires_grad=True)

        optimizer = optim.Adam([param], lr=0.1)

        for i in range(3):
            param.grad = Tensor([0.5])
            optimizer.step()

        # Parameter should decrease over multiple steps
        assert param.tolist()[0] < 5.0

    def test_adam_zero_grad(self):
        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters())

        # Set gradients
        for param in model.parameters():
            param.grad = Tensor.ones(param.shape)

        optimizer.zero_grad()

        for param in model.parameters():
            assert param.grad is None


class TestOptimizerEdgeCases:
    """Test optimizer edge cases."""

    def test_optimizer_empty_params(self):
        optimizer = optim.SGD([], lr=0.01)
        # Should not crash
        optimizer.step()
        optimizer.zero_grad()

    def test_sgd_step_with_no_params(self):
        opt = optim.SGD(iter([]), lr=0.01)
        opt.step()

    def test_adam_step_with_no_params(self):
        opt = optim.Adam(iter([]), lr=0.01)
        opt.step()

    def test_sgd_zero_lr(self):
        param = Tensor([2.0], requires_grad=True)
        param.grad = Tensor([1.0])

        optimizer = optim.SGD([param], lr=0.0)
        optimizer.step()

        # Parameter should not change
        assert abs(param.tolist()[0] - 2.0) < 1e-5

    def test_adam_zero_lr(self):
        param = Tensor([2.0], requires_grad=True)
        param.grad = Tensor([1.0])

        optimizer = optim.Adam([param], lr=0.0)
        optimizer.step()

        # Parameter should not change
        assert abs(param.tolist()[0] - 2.0) < 1e-5

    def test_sgd_negative_lr(self):
        param = Tensor([2.0], requires_grad=True)
        param.grad = Tensor([1.0])

        optimizer = optim.SGD([param], lr=-0.1)
        optimizer.step()

        # Parameter should increase: 2.0 - (-0.1) * 1.0 = 2.1
        assert abs(param.tolist()[0] - 2.1) < 1e-5

    def test_momentum_bounds(self):
        # Test momentum = 0 (should behave like regular SGD)
        param1 = Tensor([2.0], requires_grad=True)
        param1.grad = Tensor([1.0])
        optimizer1 = optim.SGD([param1], lr=0.1, momentum=0.0)
        optimizer1.step()

        param2 = Tensor([2.0], requires_grad=True)
        param2.grad = Tensor([1.0])
        optimizer2 = optim.SGD([param2], lr=0.1)
        optimizer2.step()

        assert abs(param1.tolist()[0] - param2.tolist()[0]) < 1e-5

    def test_adam_betas_bounds(self):
        # Test beta values at bounds
        param = Tensor([2.0], requires_grad=True)
        param.grad = Tensor([1.0])

        optimizer = optim.Adam([param], lr=0.1, betas=(0.0, 0.0))
        optimizer.step()

        # With beta=0, bias correction should work
        assert param.tolist()[0] != 2.0


class TestTrainingLoops:
    """Test complete training loops with optimizers."""

    def test_simple_training_loop_sgd(self):
        # Simple linear regression: y = 2*x + 1
        model = nn.Linear(1, 1, bias=True)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        loss_fn = lambda pred, target: ((pred - target) ** 2).sum()

        # Training data
        x = Tensor([[1.0], [2.0], [3.0]])
        y = Tensor([[3.0], [5.0], [7.0]])  # 2*x + 1

        initial_loss = loss_fn(model(x), y).tolist()

        # Training loop
        for _ in range(100):
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward(Tensor([1.0]))
            optimizer.step()

        final_loss = loss_fn(model(x), y).tolist()
        assert final_loss < initial_loss  # Loss should decrease

    def test_simple_training_loop_adam(self):
        # Same as above but with Adam
        model = nn.Linear(1, 1, bias=True)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = lambda pred, target: ((pred - target) ** 2).sum()

        x = Tensor([[1.0], [2.0], [3.0]])
        y = Tensor([[3.0], [5.0], [7.0]])

        initial_loss = loss_fn(model(x), y).tolist()

        for _ in range(50):  # Adam should converge faster
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward(Tensor([1.0]))
            optimizer.step()

        final_loss = loss_fn(model(x), y).tolist()
        assert final_loss < initial_loss

    def test_sgd_multiple_steps_with_momentum(self):
        linear = nn.Linear(2, 2)
        opt = optim.SGD(linear.parameters(), lr=0.01, momentum=0.9)
        x = Tensor([[1, 2]], dtype='float32')
        
        for _ in range(5):
            y = linear(x)
            loss = y.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def test_adam_multiple_steps_convergence(self):
        import tensorax.functional as F
        linear = nn.Linear(2, 1)
        opt = optim.Adam(linear.parameters(), lr=0.1)
        x = Tensor([[1, 2]], dtype='float32')
        target = Tensor([[1]], dtype='float32')
        
        for _ in range(10):
            y = linear(x)
            loss = F.mse_loss(y, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def test_convergence_comparison(self):
        # Compare SGD vs Adam convergence
        def train_model(optimizer_class, **kwargs):
            model = nn.Linear(1, 1, bias=True)
            optimizer = optimizer_class(model.parameters(), **kwargs)
            loss_fn = lambda pred, target: ((pred - target) ** 2).sum()

            x = Tensor([[1.0], [2.0]])
            y = Tensor([[3.0], [5.0]])

            losses = []
            for _ in range(20):
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                losses.append(loss.tolist())
                loss.backward(Tensor([1.0]))
                optimizer.step()

            return losses

        sgd_losses = train_model(optim.SGD, lr=0.1)
        adam_losses = train_model(optim.Adam, lr=0.1)

        # Both should decrease loss
        assert sgd_losses[-1] < sgd_losses[0]
        assert adam_losses[-1] < adam_losses[0]


class TestOptimizerValidation:
    """Test optimizer input validation."""

    def test_sgd_invalid_lr(self):
        model = nn.Linear(10, 5)
        # Negative lr should work (as tested above), but let's check edge cases
        optimizer = optim.SGD(model.parameters(), lr=float('inf'))
        # Should not crash during initialization
        assert optimizer.lr == float('inf')

    def test_adam_invalid_betas(self):
        model = nn.Linear(10, 5)
        # Invalid betas should work (Adam handles edge cases)
        optimizer = optim.Adam(model.parameters(), betas=(1.5, 0.5))
        assert optimizer.beta1 == 1.5
        assert optimizer.beta2 == 0.5

    def test_adam_invalid_eps(self):
        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters(), eps=0.0)
        assert optimizer.eps == 0.0
