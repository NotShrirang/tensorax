import pytest
import numpy as np
from tensora import functional as F, Tensor


class TestActivationFunctions:
    """Test activation functions in functional API."""

    def test_relu_forward(self):
        x = Tensor([[-1, 0, 1, 2]], dtype='float32')
        y = F.relu(x)
        expected = [[0, 0, 1, 2]]
        assert y.tolist() == expected
        assert y.shape == (1, 4)

    def test_relu_backward(self):
        x = Tensor([[-1, 0, 1, 2]], dtype='float32', requires_grad=True)
        y = F.relu(x)
        y.backward(Tensor([[1, 1, 1, 1]], dtype='float32'))
        expected_grad = [[0, 0, 1, 1]]  # dReLU/dx = 1 if x > 0 else 0
        assert x.grad.tolist() == expected_grad

    def test_relu_all_negative(self):
        x = Tensor([[-2, -1]], dtype='float32')
        y = F.relu(x)
        expected = [[0, 0]]
        assert y.tolist() == expected

    def test_relu_all_positive(self):
        x = Tensor([[1, 2, 3]], dtype='float32')
        y = F.relu(x)
        expected = [[1, 2, 3]]
        assert y.tolist() == expected

    def test_sigmoid_forward(self):
        x = Tensor([[0, 1, -1]], dtype='float32')
        y = F.sigmoid(x)
        # Check that output is between 0 and 1
        values = y.tolist()[0]
        assert all(0 <= v <= 1 for v in values)
        # Check specific values
        assert abs(values[0] - 0.5) < 1e-5  # sigmoid(0) = 0.5

    def test_sigmoid_backward(self):
        x = Tensor([[0]], dtype='float32', requires_grad=True)
        y = F.sigmoid(x)
        y.backward(Tensor([[1]], dtype='float32'))
        # d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
        sigmoid_x = y.tolist()[0][0]
        expected_grad = sigmoid_x * (1 - sigmoid_x)
        assert abs(x.grad.tolist()[0][0] - expected_grad) < 1e-5

    def test_tanh_forward(self):
        x = Tensor([[0, 1, -1]], dtype='float32')
        y = F.tanh(x)
        values = y.tolist()[0]
        # Check that output is between -1 and 1
        assert all(-1 <= v <= 1 for v in values)
        assert abs(values[0]) < 1e-5  # tanh(0) ≈ 0

    def test_tanh_backward(self):
        x = Tensor([[0]], dtype='float32', requires_grad=True)
        y = F.tanh(x)
        y.backward(Tensor([[1]], dtype='float32'))
        # d(tanh)/dx = 1 - tanh(x)^2
        tanh_x = y.tolist()[0][0]
        expected_grad = 1 - tanh_x ** 2
        assert abs(x.grad.tolist()[0][0] - expected_grad) < 1e-5

    def test_softmax_forward(self):
        x = Tensor([[1, 2, 3]], dtype='float32')
        y = F.softmax(x, dim=-1)
        values = y.tolist()[0]
        # Softmax should sum to 1
        assert abs(sum(values) - 1.0) < 1e-5
        # Values should be positive
        assert all(v > 0 for v in values)
        # Should be normalized (largest input should give largest output)
        assert values[2] > values[1] > values[0]

    def test_softmax_multidimensional(self):
        x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype='float32')
        y = F.softmax(x, dim=-1)
        # Each row should sum to 1
        result = y.tolist()
        for batch in result:
            for row in batch:
                assert abs(sum(row) - 1.0) < 1e-5

    def test_softmax_different_dims(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        # Softmax along dim=0 (columns)
        y = F.softmax(x, dim=0)
        result = y.tolist()
        # Each column should sum to 1
        for col in range(3):
            col_sum = result[0][col] + result[1][col]
            assert abs(col_sum - 1.0) < 1e-5

    def test_softmax_backward(self):
        x = Tensor([[1, 2]], dtype='float32', requires_grad=True)
        y = F.softmax(x, dim=-1)
        y.backward(Tensor([[1, 1]], dtype='float32'))
        # Gradient should exist
        assert x.grad is not None


class TestLinearFunction:
    """Test linear transformation function."""

    def test_linear_with_bias(self):
        x = Tensor([[1, 2, 3]], dtype='float32')
        weight = Tensor([[1, 0, 0], [0, 1, 0]], dtype='float32')  # Identity-like
        bias = Tensor([10, 20], dtype='float32')
        y = F.linear(x, weight, bias)
        expected = [[11, 22]]  # [1+10, 2+20]
        assert y.tolist() == expected
        assert y.shape == (1, 2)

    def test_linear_without_bias(self):
        x = Tensor([[1, 2]], dtype='float32')
        weight = Tensor([[1, 0], [0, 1]], dtype='float32')
        y = F.linear(x, weight, None)
        expected = [[1, 2]]
        assert y.tolist() == expected

    def test_linear_batch(self):
        x = Tensor([[1, 2], [3, 4]], dtype='float32')  # (2, 2)
        weight = Tensor([[1, 0], [0, 1]], dtype='float32')  # (2, 2)
        bias = Tensor([1, 1], dtype='float32')
        y = F.linear(x, weight, bias)
        expected = [[2, 3], [4, 5]]  # Each row + 1
        assert y.tolist() == expected
        assert y.shape == (2, 2)

    def test_linear_backward(self):
        x = Tensor([[1, 2]], dtype='float32', requires_grad=True)
        weight = Tensor([[1, 0], [0, 1]], dtype='float32', requires_grad=True)
        bias = Tensor([0, 0], dtype='float32', requires_grad=True)
        y = F.linear(x, weight, bias)
        y.backward(Tensor([[1, 1]], dtype='float32'))
        assert x.grad is not None
        assert weight.grad is not None
        assert bias.grad is not None


class TestLossFunctions:
    """Test loss functions."""

    def test_mse_loss_zero(self):
        pred = Tensor([[1, 2, 3]], dtype='float32')
        target = Tensor([[1, 2, 3]], dtype='float32')
        loss = F.mse_loss(pred, target)
        assert loss.tolist() == 0  # Perfect prediction

    def test_mse_loss_nonzero(self):
        pred = Tensor([[1, 2, 3]], dtype='float32')
        target = Tensor([[2, 3, 4]], dtype='float32')
        loss = F.mse_loss(pred, target)
        # MSE = ((1-2)^2 + (2-3)^2 + (3-4)^2) / 3 = (1 + 1 + 1) / 3 = 1
        assert abs(loss.tolist() - 1.0) < 1e-5

    def test_mse_loss_multidimensional(self):
        pred = Tensor([[[1, 2]], [[3, 4]]], dtype='float32')
        target = Tensor([[[2, 3]], [[4, 5]]], dtype='float32')
        loss = F.mse_loss(pred, target)
        # MSE = mean of ((1-2)^2 + (2-3)^2 + (3-4)^2 + (4-5)^2) = mean(1+1+1+1) = 1
        assert abs(loss.tolist() - 1.0) < 1e-5

    def test_mse_loss_backward(self):
        pred = Tensor([[1, 2]], dtype='float32', requires_grad=True)
        target = Tensor([[2, 3]], dtype='float32')
        loss = F.mse_loss(pred, target)
        loss.backward()
        # d(MSE)/d(pred) = 2 * (pred - target) / n
        # For pred=[1,2], target=[2,3], n=2
        # grad = 2 * ([1-2, 2-3]) / 2 = [-2, -2]/1 = [-1, -1]
        expected_grad = [[-1, -1]]
        assert pred.grad.tolist() == expected_grad

    def test_cross_entropy_loss(self):
        # This will fail until implemented
        pred = Tensor([[1, 2, 3]], dtype='float32')
        target = Tensor([[0, 0, 1]], dtype='float32')
        with pytest.raises(RuntimeError):
            F.cross_entropy_loss(pred, target)


class TestUnimplementedFunctions:
    """Test functions that are not yet implemented."""

    def test_conv2d_not_implemented(self):
        x = Tensor(np.random.randn(1, 3, 28, 28), dtype='float32')
        weight = Tensor(np.random.randn(32, 3, 3, 3), dtype='float32')
        with pytest.raises(NotImplementedError):
            F.conv2d(x, weight)

    def test_max_pool2d_not_implemented(self):
        x = Tensor(np.random.randn(1, 3, 28, 28), dtype='float32')
        with pytest.raises(NotImplementedError):
            F.max_pool2d(x, kernel_size=2)


class TestFunctionalEdgeCases:
    """Test edge cases for functional operations."""

    def test_activations_on_empty_tensor(self):
        x = Tensor([], dtype='float32')
        y = F.relu(x)
        assert y.shape == (0,)

    def test_activations_on_scalar(self):
        x = Tensor([5], dtype='float32')
        y = F.relu(x)
        assert y.tolist() == [5]

    def test_softmax_single_element(self):
        x = Tensor([[1]], dtype='float32')
        y = F.softmax(x, dim=-1)
        assert abs(y.tolist()[0][0] - 1.0) < 1e-5

    def test_linear_single_feature(self):
        x = Tensor([[1]], dtype='float32')
        weight = Tensor([[2]], dtype='float32')
        bias = Tensor([3], dtype='float32')
        y = F.linear(x, weight, bias)
        assert y.tolist() == [[5]]  # 1*2 + 3

    def test_mse_loss_different_shapes(self):
        pred = Tensor([[1, 2]], dtype='float32')
        target = Tensor([[1]], dtype='float32')
        # Should handle broadcasting or raise error
        with pytest.raises(RuntimeError):
            F.mse_loss(pred, target)

    def test_gradient_through_multiple_functions(self):
        x = Tensor([[1, 2]], dtype='float32', requires_grad=True)
        y = F.relu(x)
        z = F.sigmoid(y)
        z.backward(Tensor([[1, 1]], dtype='float32'))
        assert x.grad is not None
