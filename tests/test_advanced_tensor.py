import pytest
import numpy as np
from tensorax import Tensor


class TestAdvancedTensorOps:
    """Test advanced tensor operations that may not be implemented yet."""

    def test_tensor_reshape(self):
        # This will likely fail until reshape is implemented
        x = Tensor([[1, 2, 3, 4]], dtype='float32')
        with pytest.raises(AttributeError):
            y = x.reshape((2, 2))
            assert y.shape == (2, 2)

    def test_tensor_view(self):
        # View operation
        x = Tensor([[1, 2, 3, 4]], dtype='float32')
        with pytest.raises(AttributeError):
            y = x.view((2, 2))
            assert y.shape == (2, 2)

    def test_tensor_squeeze(self):
        # Remove dimensions of size 1
        x = Tensor([[[1], [2]]], dtype='float32')  # (1, 2, 1)
        with pytest.raises(AttributeError):
            y = x.squeeze()
            assert y.shape == (2,)

    def test_tensor_unsqueeze(self):
        # Add dimension of size 1
        x = Tensor([[1, 2]], dtype='float32')  # (2, 2)
        with pytest.raises(AttributeError):
            y = x.unsqueeze(dim=0)
            assert y.shape == (1, 2, 2)

    def test_tensor_sum(self):
        # Sum along dimensions
        x = Tensor([[1, 2], [3, 4]], dtype='float32')
        y = x.sum(dim=0)
        assert y.tolist() == [4, 6]

    def test_tensor_mean(self):
        # Mean along dimensions
        x = Tensor([[1, 2], [3, 4]], dtype='float32')
        y = x.mean(dim=0)
        assert y.tolist() == [2, 3]

    def test_tensor_max(self):
        # Max along dimensions
        x = Tensor([[1, 5], [3, 2]], dtype='float32')
        with pytest.raises(AttributeError):
            y = x.max(dim=0)
            assert y.tolist() == [3, 5]

    def test_tensor_min(self):
        # Min along dimensions
        x = Tensor([[1, 5], [3, 2]], dtype='float32')
        with pytest.raises(AttributeError):
            y = x.min(dim=0)
            assert y.tolist() == [1, 2]

    def test_tensor_argmax(self):
        # Argmax along dimensions
        x = Tensor([[1, 5], [3, 2]], dtype='float32')
        with pytest.raises(AttributeError):
            y = x.argmax(dim=0)
            assert y.tolist() == [1, 0]

    def test_tensor_argmin(self):
        # Argmin along dimensions
        x = Tensor([[1, 5], [3, 2]], dtype='float32')
        with pytest.raises(AttributeError):
            y = x.argmin(dim=0)
            assert y.tolist() == [0, 1]

    def test_tensor_exp(self):
        # Element-wise exponential
        x = Tensor([[0, 1]], dtype='float32')
        y = x.exp()
        assert abs(y.tolist()[0][0] - 1.0) < 1e-5  # exp(0) = 1
        assert abs(y.tolist()[0][1] - np.e) < 1e-5  # exp(1) = e

    def test_tensor_log(self):
        # Element-wise natural log
        x = Tensor([[1, np.e]], dtype='float32')
        y = x.log()
        assert abs(y.tolist()[0][0]) < 1e-5  # log(1) = 0
        assert abs(y.tolist()[0][1] - 1.0) < 1e-5  # log(e) = 1

    def test_tensor_pow(self):
        # Element-wise power
        x = Tensor([[2, 3]], dtype='float32')
        with pytest.raises(AttributeError):
            y = x.pow(2)
            assert y.tolist() == [[4, 9]]

    def test_tensor_abs(self):
        # Element-wise absolute value
        x = Tensor([[-1, 2, -3]], dtype='float32')
        with pytest.raises(AttributeError):
            y = x.abs()
            assert y.tolist() == [[1, 2, 3]]

    def test_tensor_clamp(self):
        # Clamp values to range
        x = Tensor([[0, 2, 5, -1]], dtype='float32')
        with pytest.raises(AttributeError):
            y = x.clamp(min=0, max=3)
            assert y.tolist() == [[0, 2, 3, 0]]


class TestTensorIndexing:
    """Test tensor indexing operations."""

    def test_tensor_getitem_single_element(self):
        x = Tensor([[1, 2], [3, 4]], dtype='float32')
        with pytest.raises(TypeError):
            # Should return scalar or 0-d tensor
            y = x[0, 1]
            assert y == 2

    def test_tensor_getitem_row(self):
        x = Tensor([[1, 2], [3, 4]], dtype='float32')
        with pytest.raises(TypeError):
            y = x[0]
            assert isinstance(y, Tensor)
            assert y.tolist() == [1, 2]

    def test_tensor_getitem_slice(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        with pytest.raises(TypeError):
            y = x[:, 1:]
            assert y.tolist() == [[2, 3], [5, 6]]

    def test_tensor_setitem(self):
        x = Tensor([[1, 2], [3, 4]], dtype='float32')
        with pytest.raises(TypeError):
            x[0, 1] = 10
            assert x.tolist() == [[1, 10], [3, 4]]


class TestTensorComparison:
    """Test tensor comparison operations."""

    def test_tensor_eq(self):
        x = Tensor([[1, 2], [3, 4]], dtype='float32')
        y = Tensor([[1, 3], [3, 4]], dtype='float32')
        with pytest.raises(AttributeError):
            z = x.eq(y)
            assert z.tolist() == [[True, False], [True, True]]

    def test_tensor_lt(self):
        x = Tensor([[1, 2], [3, 4]], dtype='float32')
        y = Tensor([[2, 1], [4, 3]], dtype='float32')
        with pytest.raises(AttributeError):
            z = x.lt(y)
            assert z.tolist() == [[True, False], [True, False]]

    def test_tensor_gt(self):
        x = Tensor([[1, 2], [3, 4]], dtype='float32')
        y = Tensor([[0, 3], [2, 5]], dtype='float32')
        with pytest.raises(AttributeError):
            z = x.gt(y)
            assert z.tolist() == [[True, False], [True, False]]


class TestBroadcasting:
    """Test broadcasting operations."""

    def test_broadcasting_scalars(self):
        x = Tensor([[1, 2]], dtype='float32')  # (1, 2)
        y = Tensor([[[3], [4]], [[5], [6]]], dtype='float32')  # (2, 2, 1)
        # Broadcasting should work for compatible shapes
        # with pytest.raises(RuntimeError):
        z = x + y  # Should broadcast to (2, 2, 2)
        expected = [[[4.0, 5.0], [5.0, 6.0]], [[6.0, 7.0], [7.0, 8.0]]]
        assert z.tolist() == expected

    def test_broadcasting_incompatible(self):
        x = Tensor([[1, 2, 3]], dtype='float32')  # (1, 3)
        y = Tensor([[1, 2]], dtype='float32')  # (1, 2)
        with pytest.raises(RuntimeError):
            z = x + y  # Incompatible for broadcasting


class TestTensorSerialization:
    """Test tensor save/load operations."""

    def test_tensor_save(self):
        x = Tensor([[1, 2], [3, 4]], dtype='float32')
        with pytest.raises(AttributeError):
            x.save('test_tensor.npy')

    def test_tensor_load(self):
        with pytest.raises(AttributeError):
            x = Tensor.load('test_tensor.npy')
            assert x.shape == (2, 2)


class TestMemoryManagement:
    """Test memory management and cleanup."""

    def test_tensor_del(self):
        x = Tensor([[1, 2], [3, 4]], dtype='float32')
        del x
        # Should not crash

    # def test_large_tensor_creation(self):
    #     # Test creating very large tensors
    #     with pytest.raises(RuntimeError):
    #         # This might fail due to memory limits
    #         x = Tensor.zeros((10000, 10000), dtype='float32')

    def test_tensor_memory_sharing(self):
        # Test if operations create copies or share memory
        x = Tensor([[1, 2], [3, 4]], dtype='float32')
        with pytest.raises(AttributeError):
            y = x.detach()  # Should create a copy without gradients
            assert y.tolist() == x.tolist()
            assert not y.requires_grad