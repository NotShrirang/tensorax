import pytest
import numpy as np
from tensora import Tensor, cuda_is_available


class TestTensorCreation:
    """Test tensor creation with various inputs and parameters."""

    def test_tensor_from_list_1d(self):
        data = [1, 2, 3, 4]
        tensor = Tensor(data)
        assert tensor.shape == (4,)
        assert tensor.dtype == 'float32'
        assert tensor.device == 'cpu'
        assert not tensor.requires_grad

    def test_tensor_from_list_2d(self):
        data = [[1, 2], [3, 4]]
        tensor = Tensor(data)
        assert tensor.shape == (2, 2)

    def test_tensor_from_list_3d(self):
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        tensor = Tensor(data)
        assert tensor.shape == (2, 2, 2)

    def test_tensor_from_nested_list_irregular(self):
        # Should handle irregular shapes gracefully or raise appropriate error
        with pytest.raises(ValueError):
            Tensor([[1, 2], [3]])  # Irregular shape

    def test_tensor_from_empty_list(self):
        tensor = Tensor([])
        assert tensor.shape == (0,)

    def test_tensor_from_scalar_list(self):
        tensor = Tensor([5])
        assert tensor.shape == (1,)

    def test_tensor_with_explicit_shape(self):
        data = [1, 2, 3, 4]
        tensor = Tensor(data, shape=(2, 2))
        assert tensor.shape == (2, 2)

    def test_tensor_shape_mismatch(self):
        data = [1, 2, 3]
        with pytest.raises(ValueError):
            Tensor(data, shape=(2, 2))  # 3 elements != 4 expected

    def test_tensor_dtypes(self):
        for dtype in ['float32', 'float64', 'int32', 'int64']:
            tensor = Tensor([1, 2, 3], dtype=dtype)
            assert tensor.dtype == dtype

    def test_tensor_invalid_dtype(self):
        with pytest.raises(ValueError):
            Tensor([1, 2, 3], dtype='invalid')

    def test_tensor_requires_grad(self):
        tensor = Tensor([1, 2, 3], requires_grad=True)
        assert tensor.requires_grad
        assert tensor.grad is None

    def test_tensor_copy_from_tensor(self):
        original = Tensor([[1, 2], [3, 4]], requires_grad=True)
        copy = Tensor(original)
        assert copy.shape == original.shape
        assert copy.dtype == original.dtype
        assert copy.requires_grad == original.requires_grad

    @pytest.mark.skipif(not cuda_is_available(), reason="CUDA not available")
    def test_tensor_cuda_device(self):
        tensor = Tensor([1, 2, 3], device='cuda')
        assert tensor.device == 'cuda'

    def test_tensor_invalid_device(self):
        with pytest.raises(ValueError):
            Tensor([1, 2, 3], device='invalid')

    def test_tensor_zeros_static(self):
        tensor = Tensor.zeros((2, 3))
        assert tensor.shape == (2, 3)
        assert np.allclose(tensor.tolist(), [[0, 0, 0], [0, 0, 0]])

    def test_tensor_ones_static(self):
        tensor = Tensor.ones((2, 3))
        assert tensor.shape == (2, 3)
        assert np.allclose(tensor.tolist(), [[1, 1, 1], [1, 1, 1]])

    def test_tensor_full_static(self):
        tensor = Tensor.full((2, 3), 5.5)
        assert tensor.shape == (2, 3)
        assert np.allclose(tensor.tolist(), [[5.5, 5.5, 5.5], [5.5, 5.5, 5.5]])

    def test_tensor_randn_static(self):
        tensor = Tensor.randn((2, 3))
        assert tensor.shape == (2, 3)


class TestTensorProperties:
    """Test tensor properties and methods."""

    def test_shape_property(self):
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])
        assert tensor.shape == (2, 3)

    def test_ndim_property(self):
        assert Tensor([1]).ndim == 1
        assert Tensor([[1, 2]]).ndim == 2
        assert Tensor([[[1]]]).ndim == 3

    def test_size_property(self):
        assert Tensor([1, 2, 3]).size == 3
        assert Tensor([[1, 2], [3, 4]]).size == 4

    def test_tolist_method(self):
        data = [[1, 2], [3, 4]]
        tensor = Tensor(data)
        result = tensor.tolist()
        assert result == data

    def test_repr(self):
        tensor = Tensor([[1, 2], [3, 4]])
        repr_str = repr(tensor)
        assert 'Tensor' in repr_str
        assert '(2, 2)' in repr_str

    def test_str(self):
        tensor = Tensor([[1, 2], [3, 4]])
        str_repr = str(tensor)
        assert 'Tensor' in str_repr


class TestTensorOperations:
    """Test tensor arithmetic operations."""

    def test_addition_tensor_tensor(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = a + b
        expected = [[6, 8], [10, 12]]
        assert c.tolist() == expected
        assert c.shape == (2, 2)

    def test_addition_tensor_scalar(self):
        a = Tensor([[1, 2], [3, 4]])
        c = a + 10
        expected = [[11, 12], [13, 14]]
        assert c.tolist() == expected

    def test_addition_scalar_tensor(self):
        a = Tensor([[1, 2], [3, 4]])
        c = 10 + a
        expected = [[11, 12], [13, 14]]
        assert c.tolist() == expected

    def test_addition_broadcasting(self):
        a = Tensor([[1, 2, 3]])  # (1, 3)
        b = Tensor([[4], [5]])   # (2, 1)
        c = a + b
        expected = [[5, 6, 7], [6, 7, 8]]
        assert c.tolist() == expected
        assert c.shape == (2, 3)

    def test_addition_device_mismatch(self):
        a = Tensor([1, 2], device='cpu')
        # This will fail if CUDA is available, but test the logic
        if cuda_is_available():
            b = Tensor([3, 4], device='cuda')
            with pytest.raises(RuntimeError):
                a + b

    def test_subtraction(self):
        a = Tensor([[5, 6], [7, 8]])
        b = Tensor([[1, 2], [3, 4]])
        c = a - b
        expected = [[4, 4], [4, 4]]
        assert c.tolist() == expected

    def test_multiplication(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[2, 0], [0, 2]])
        c = a * b
        expected = [[2, 0], [0, 8]]
        assert c.tolist() == expected

    def test_division(self):
        a = Tensor([[4, 6], [8, 10]])
        b = Tensor([[2, 3], [4, 5]])
        c = a / b
        expected = [[2, 2], [2, 2]]
        assert c.tolist() == expected

    def test_matmul_2d_2d(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = a @ b
        expected = [[19, 22], [43, 50]]
        assert c.tolist() == expected
        assert c.shape == (2, 2)

    def test_matmul_1d_1d(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        with pytest.raises(RuntimeError):
            a @ b # 1D @ 1D is not a valid operation in this implementation 

    def test_matmul_incompatible_shapes(self):
        a = Tensor([[1, 2, 3]])  # (1, 3)
        b = Tensor([[1, 2], [3, 4]])  # (2, 2)
        with pytest.raises(RuntimeError):
            a @ b

    def test_matmul_3d_tensors(self):
        # Batch matrix multiplication
        a = Tensor([[[1, 2]], [[3, 4]]])  # (2, 1, 2)
        b = Tensor([[[5], [6]], [[7], [8]]])  # (2, 2, 1)
        c = a @ b
        expected = [[[17.]], [[53.]]]
        assert c.tolist() == expected
        assert c.shape == (2, 1, 1)

    def test_transpose_2d(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        b = a.T
        expected = [[1, 4], [2, 5], [3, 6]]
        assert b.tolist() == expected
        assert b.shape == (3, 2)

    def test_transpose_1d(self):
        a = Tensor([1, 2, 3])
        b = a.T
        assert b.tolist() == [1, 2, 3]  # 1D transpose is identity
        assert b.shape == (3,)

    def test_sqrt(self):
        a = Tensor([1, 4, 9, 16])
        b = a.sqrt()
        expected = [1, 2, 3, 4]
        assert b.tolist() == expected


class TestTensorDeviceOperations:
    """Test device-related operations."""

    @pytest.mark.skipif(not cuda_is_available(), reason="CUDA not available")
    def test_cpu_to_cuda(self):
        a = Tensor([[1, 2], [3, 4]])
        b = a.cuda()
        assert b.device == 'cuda'
        assert a.device == 'cpu'  # Original unchanged

    @pytest.mark.skipif(not cuda_is_available(), reason="CUDA not available")
    def test_cuda_to_cpu(self):
        a = Tensor([[1, 2], [3, 4]], device='cuda')
        b = a.cpu()
        assert b.device == 'cpu'
        assert a.device == 'cuda'  # Original unchanged

    def test_to_method(self):
        a = Tensor([[1, 2], [3, 4]])
        b = a.to('cpu')
        assert b.device == 'cpu'

    def test_to_invalid_device(self):
        a = Tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            a.to('invalid')


class TestTensorIteration:
    """Test tensor iteration capabilities."""

    def test_iterate_1d_tensor(self):
        tensor = Tensor([1, 2, 3, 4])
        values = list(tensor)
        assert values == [1, 2, 3, 4]

    def test_iterate_2d_tensor(self):
        tensor = Tensor([[1, 2], [3, 4]])
        rows = list(tensor)
        assert len(rows) == 2
        assert rows[0].tolist() == [1, 2]
        assert rows[1].tolist() == [3, 4]

    def test_len_1d(self):
        tensor = Tensor([1, 2, 3])
        assert len(tensor) == 3

    def test_len_2d(self):
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])
        assert len(tensor) == 2

    def test_len_0d_error(self):
        tensor = Tensor(5)  # Scalar
        with pytest.raises(TypeError):
            len(tensor)

    def test_iterate_0d_error(self):
        tensor = Tensor(5)  # Scalar
        with pytest.raises(TypeError):
            list(tensor)


class TestGradientComputation:
    """Test automatic differentiation."""

    def test_simple_addition_backward(self):
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        c = a + b
        c.backward()
        assert a.grad.tolist() == [1.0]
        assert b.grad.tolist() == [1.0]

    def test_multiplication_backward(self):
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        c = a * b
        c.backward()
        assert a.grad.tolist() == [3.0]
        assert b.grad.tolist() == [2.0]

    def test_subtraction_backward(self):
        a = Tensor([5.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        c = a - b
        c.backward()
        assert a.grad.tolist() == [1.0]
        assert b.grad.tolist() == [-1.0]

    def test_division_backward(self):
        a = Tensor([6.0], requires_grad=True)
        b = Tensor([2.0], requires_grad=True)
        c = a / b
        c.backward()
        assert a.grad.tolist() == [0.5]  # 1/b
        assert b.grad.tolist() == [-1.5]  # -a/b^2

    def test_matmul_backward(self):
        a = Tensor([[1.0, 2.0]], requires_grad=True)  # (1, 2)
        b = Tensor([[3.0], [4.0]], requires_grad=True)  # (2, 1)
        c = a @ b  # (1, 1)
        c.backward()
        assert a.grad.tolist() == [[3.0, 4.0]]
        assert b.grad.tolist() == [[1.0], [2.0]]

    def test_transpose_backward(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = a.T
        b.backward(Tensor([[1.0, 1.0], [1.0, 1.0]]))
        assert a.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]

    def test_sqrt_backward(self):
        a = Tensor([4.0], requires_grad=True)
        b = a.sqrt()
        b.backward(Tensor([1.0]))
        assert a.grad.tolist() == [0.25]  # 1/(2*sqrt(4)) = 1/(2*2) = 0.25

    def test_chained_gradients(self):
        a = Tensor([2.0], requires_grad=True)
        b = a * a  # b = a^2
        c = b + a  # c = a^2 + a
        c.backward()
        assert a.grad.tolist() == [5.0]  # dc/da = 2a + 1 = 2*2 + 1 = 5

    def test_no_grad_context(self):
        # Test that operations don't track gradients when requires_grad=False
        a = Tensor([2.0], requires_grad=False)
        b = Tensor([3.0], requires_grad=False)
        c = a + b
        assert not c.requires_grad
        assert c._grad_fn is None

    def test_mixed_grad_requirements(self):
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=False)
        c = a + b
        assert c.requires_grad
        c.backward()
        assert a.grad.tolist() == [1.0]
        # b doesn't require grad, so no gradient computed

    def test_zero_grad(self):
        a = Tensor([2.0], requires_grad=True)
        b = a * a
        b.backward()
        assert a.grad is not None
        a.zero_grad()
        assert a.grad is None

    def test_backward_with_custom_grad(self):
        a = Tensor([2.0], requires_grad=True)
        b = a * a
        b.backward(Tensor([0.5]))  # Custom gradient
        assert a.grad.tolist() == [2.0]  # 2 * a * 0.5 = 2 * 2 * 0.5 = 2.0

    def test_scalar_backward_no_grad_specified(self):
        a = Tensor([2.0], requires_grad=True)
        b = a * a
        b.backward()  # Should work for scalar tensors
        assert a.grad.tolist() == [4.0]  # 2 * a = 2 * 2 = 4.0

    def test_non_scalar_backward_requires_grad(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = a * a
        with pytest.raises(RuntimeError):
            b.backward()  # Non-scalar needs explicit grad


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_tensor_operations(self):
        a = Tensor([])
        b = Tensor([])
        c = a + b
        assert c.shape == (0,)

    def test_single_element_tensor(self):
        a = Tensor([5])
        assert a.shape == (1,)
        assert a.tolist() == [5]

    def test_large_tensor_creation(self):
        # Test with reasonably large tensor
        size = 1000
        data = list(range(size))
        tensor = Tensor(data)
        assert tensor.shape == (size,)
        assert tensor.size == size

    def test_deeply_nested_list(self):
        # Test with deeply nested structure
        data = [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]]
        tensor = Tensor(data)
        assert tensor.shape == (2, 2, 1, 2)

    def test_tensor_with_floats_and_ints(self):
        data = [1, 2.5, 3, 4.0]
        tensor = Tensor(data)
        assert tensor.dtype == 'float32'  # Should convert to float

    def test_negative_numbers(self):
        data = [-1, -2, 3, -4]
        tensor = Tensor(data)
        result = tensor.tolist()
        assert result == [-1, -2, 3, -4]

    def test_zero_dimensional_tensor(self):
        # Scalar tensor
        tensor = Tensor(5)
        assert tensor.shape == ()
        assert tensor.size == 1
        assert tensor.tolist() == 5

    def test_gradient_through_multiple_operations(self):
        # Complex computation graph
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        c = a * b + a  # c = a*b + a
        d = c * c  # d = c^2
        d.backward()
        # dd/da = 2*c * dc/da = 2*c * (b + 1)
        # c = 2*3 + 2 = 8, dc/da = 3 + 1 = 4, dd/da = 2*8*4 = 64
        expected_a_grad = 2 * (a.tolist()[0] * b.tolist()[0] + a.tolist()[0]) * (b.tolist()[0] + 1)
        assert abs(a.grad.tolist()[0] - expected_a_grad) < 1e-5
