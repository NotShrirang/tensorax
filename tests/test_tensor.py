import pytest
import numpy as np
from tensorax import Tensor, cuda_is_available
from tensorax.utils.shape_utils import _compute_size, _has_valid_shape, _infer_shape


class TestShapeUtils:
    """Test coverage for shape_utils.py functions."""

    def test_compute_size_empty_shape(self):
        assert _compute_size(()) == 1

    def test_compute_size_1d(self):
        assert _compute_size((5,)) == 5

    def test_compute_size_2d(self):
        assert _compute_size((3, 4)) == 12

    def test_compute_size_3d(self):
        assert _compute_size((2, 3, 4)) == 24

    def test_has_valid_shape_empty_list(self):
        assert _has_valid_shape([]) is True

    def test_has_valid_shape_single_number(self):
        assert _has_valid_shape(5) is True
        assert _has_valid_shape(5.5) is True

    def test_has_valid_shape_flat_list(self):
        assert _has_valid_shape([1, 2, 3]) is True

    def test_has_valid_shape_nested_valid(self):
        assert _has_valid_shape([[1, 2], [3, 4]]) is True

    def test_has_valid_shape_nested_invalid(self):
        assert _has_valid_shape([[1, 2], [3]]) is False

    def test_has_valid_shape_mixed_types_in_flat(self):
        assert _has_valid_shape([1, 2.5, 3]) is True

    def test_has_valid_shape_inconsistent_nested(self):
        assert _has_valid_shape([1, [2, 3]]) is False

    def test_has_valid_shape_tensor_input(self):
        t = Tensor([1, 2, 3])
        assert _has_valid_shape(t) is True

    def test_has_valid_shape_numpy_input(self):
        arr = np.array([1, 2, 3])
        assert _has_valid_shape(arr) is True

    def test_infer_shape_flat_list(self):
        assert _infer_shape([1, 2, 3]) == (3,)

    def test_infer_shape_2d(self):
        assert _infer_shape([[1, 2], [3, 4]]) == (2, 2)

    def test_infer_shape_3d(self):
        assert _infer_shape([[[1], [2]], [[3], [4]]]) == (2, 2, 1)

    def test_infer_shape_empty_list(self):
        assert _infer_shape([]) == (0,)

    def test_infer_shape_empty_nested(self):
        assert _infer_shape([[]]) == (1, 0)


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

    def test_zeros_with_requires_grad(self):
        t = Tensor.zeros((2, 3), requires_grad=True)
        assert t.requires_grad
        assert t.shape == (2, 3)

    def test_ones_with_device(self):
        t = Tensor.ones((2, 3), device='cpu')
        assert t.device == 'cpu'

    def test_full_with_dtype(self):
        t = Tensor.full((2, 2), 3.14, dtype='float64')
        assert t.dtype == 'float64'

    def test_randn_with_all_params(self):
        t = Tensor.randn((3, 3), dtype='float32', device='cpu', requires_grad=True)
        assert t.shape == (3, 3)
        assert t.requires_grad

    def test_copy_preserves_grad_requirement(self):
        a = Tensor([1, 2, 3], dtype='float32', requires_grad=True)
        b = Tensor(a)
        assert b.requires_grad == True

    def test_copy_preserves_dtype(self):
        a = Tensor([1, 2, 3], dtype='float64')
        b = Tensor(a)
        assert b.dtype == 'float64'


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
        assert 'device=' in repr_str
        assert 'dtype=' in repr_str

    def test_str(self):
        tensor = Tensor([[1, 2], [3, 4]])
        str_repr = str(tensor)
        assert 'Tensor' in str_repr
        assert 'device=' in str_repr
        assert 'dtype=' in str_repr


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

    def test_rmul_scalar(self):
        a = Tensor([1, 2, 3], dtype='float32')
        c = 2 * a  # Uses __rmul__
        assert np.allclose(c.tolist(), [2, 4, 6])

    def test_rmul_float(self):
        a = Tensor([1, 2, 3], dtype='float32')
        c = 2.5 * a
        assert np.allclose(c.tolist(), [2.5, 5.0, 7.5])

    def test_rsub_scalar(self):
        a = Tensor([1, 2, 3], dtype='float32')
        with pytest.raises(TypeError):
            c = 10 - a  # __rsub__ not implemented

    def test_rtruediv_scalar(self):
        a = Tensor([1, 2, 4], dtype='float32')
        with pytest.raises(TypeError):
            c = 8 / a  # __rtruediv__ not implemented

    def test_neg_operator(self):
        a = Tensor([1, -2, 3], dtype='float32')
        with pytest.raises(TypeError):
            b = -a  # __neg__ not implemented

    def test_division_by_small_number(self):
        a = Tensor([1, 2, 3], dtype='float32')
        b = Tensor([0.001, 0.001, 0.001], dtype='float32')
        c = a / b
        assert np.allclose(c.tolist(), [1000, 2000, 3000])

    def test_pow_square(self):
        a = Tensor([2, 3, 4], dtype='float32')
        b = a ** 2
        assert np.allclose(b.tolist(), [4, 9, 16])

    def test_pow_fractional(self):
        a = Tensor([4, 9, 16], dtype='float32')
        b = a ** 0.5
        assert np.allclose(b.tolist(), [2, 3, 4])

    def test_pow_zero(self):
        a = Tensor([2, 3, 4], dtype='float32')
        b = a ** 0
        assert np.allclose(b.tolist(), [1, 1, 1])

    def test_pow_negative(self):
        a = Tensor([2, 4], dtype='float32')
        b = a ** -1
        assert np.allclose(b.tolist(), [0.5, 0.25])

    def test_pow_with_gradient(self):
        a = Tensor([2, 3], dtype='float32', requires_grad=True)
        b = a ** 2
        assert b.requires_grad
        b.sum().backward()
        assert a.grad is not None

    def test_sqrt_with_gradient(self):
        a = Tensor([4, 9], dtype='float32', requires_grad=True)
        b = a.sqrt()
        assert b.requires_grad
        b.sum().backward()
        assert a.grad is not None

    def test_transpose_3d(self):
        a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype='float32')
        b = a.T
        assert b.shape == (2, 2, 2)

    def test_transpose_with_grad(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32', requires_grad=True)
        b = a.T
        assert b.requires_grad
        assert b._grad_fn[0] == 'transpose'


class TestMatmulOperations:
    """Test matrix multiplication operations and edge cases."""

    def test_matmul_basic(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        b = Tensor([[5, 6], [7, 8]], dtype='float32')
        c = a.matmul(b)
        expected = [[19, 22], [43, 50]]
        assert np.allclose(c.tolist(), expected)

    def test_matmul_with_default_method(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        b = Tensor([[1, 0], [0, 1]], dtype='float32')
        c = a.matmul(b, method='default')
        assert np.allclose(c.tolist(), [[1, 2], [3, 4]])

    def test_matmul_incompatible_shapes(self):
        a = Tensor([[1, 2, 3]], dtype='float32')
        b = Tensor([[1, 2], [3, 4]], dtype='float32')
        with pytest.raises(RuntimeError):
            a.matmul(b)

    def test_matmul_1d_raises_error(self):
        a = Tensor([1, 2, 3], dtype='float32')
        b = Tensor([4, 5, 6], dtype='float32')
        with pytest.raises(RuntimeError):
            a.matmul(b)

    def test_matmul_backward_propagates(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32', requires_grad=True)
        b = Tensor([[1, 0], [0, 1]], dtype='float32', requires_grad=True)
        c = a.matmul(b)
        assert c.requires_grad
        assert c._grad_fn is not None
        assert c._grad_fn[0] == 'matmul'


class TestSumMeanOperations:
    """Test sum and mean operations with various dimensions."""

    def test_sum_all_elements(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        s = a.sum()
        assert s.shape == ()
        assert np.isclose(s.tolist(), 10)

    def test_sum_dim0(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        s = a.sum(dim=0)
        assert s.shape == (2,)
        assert np.allclose(s.tolist(), [4, 6])

    def test_sum_dim1(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        s = a.sum(dim=1)
        assert s.shape == (2,)
        assert np.allclose(s.tolist(), [3, 7])

    def test_sum_invalid_dim(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        with pytest.raises(ValueError):
            a.sum(dim=5)

    def test_sum_negative_dim_invalid(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        with pytest.raises(ValueError):
            a.sum(dim=-1)

    def test_mean_all_elements(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        m = a.mean()
        assert m.shape == ()
        assert np.isclose(m.tolist(), 2.5)

    def test_mean_dim0(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        m = a.mean(dim=0)
        assert np.allclose(m.tolist(), [2, 3])

    def test_mean_invalid_dim(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        with pytest.raises(ValueError):
            a.mean(dim=10)

    def test_sum_with_gradient(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32', requires_grad=True)
        s = a.sum()
        assert s.requires_grad
        s.backward()
        assert a.grad is not None

    def test_mean_with_gradient(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32', requires_grad=True)
        m = a.mean()
        assert m.requires_grad
        m.backward()
        assert a.grad is not None


class TestExpLogOperations:
    """Test exp and log operations with gradient tracking."""

    def test_exp_basic(self):
        a = Tensor([0, 1, 2], dtype='float32')
        b = a.exp()
        assert np.isclose(b.tolist()[0], 1.0, rtol=1e-5)
        assert np.isclose(b.tolist()[1], np.e, rtol=1e-5)

    def test_exp_with_gradient(self):
        a = Tensor([0, 1], dtype='float32', requires_grad=True)
        b = a.exp()
        assert b.requires_grad
        assert b._grad_fn[0] == 'exp'

    def test_log_basic(self):
        a = Tensor([1, np.e, np.e**2], dtype='float32')
        b = a.log()
        assert np.isclose(b.tolist()[0], 0.0, atol=1e-5)
        assert np.isclose(b.tolist()[1], 1.0, rtol=1e-5)
        assert np.isclose(b.tolist()[2], 2.0, rtol=1e-5)

    def test_log_with_gradient(self):
        a = Tensor([1, 2], dtype='float32', requires_grad=True)
        b = a.log()
        assert b.requires_grad
        assert b._grad_fn[0] == 'log'


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

    def test_cpu_to_cpu_returns_same(self):
        a = Tensor([1, 2, 3], dtype='float32', device='cpu')
        b = a.cpu()
        assert b.device == 'cpu'
        assert np.allclose(b.tolist(), [1, 2, 3])

    @pytest.mark.skipif(not cuda_is_available(), reason="CUDA not available")
    def test_cuda_to_cuda_returns_same(self):
        a = Tensor([1, 2, 3], dtype='float32', device='cuda')
        b = a.cuda()
        assert b.device == 'cuda'


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

    def test_iterate_0d_via_sum(self):
        a = Tensor([5], dtype='float32')
        scalar = a.sum()
        assert scalar.shape == ()
        with pytest.raises(TypeError):
            for _ in scalar:
                pass

    def test_len_0d_via_sum(self):
        a = Tensor([5], dtype='float32')
        scalar = a.sum()
        with pytest.raises(TypeError):
            len(scalar)


class TestTensorEquality:
    """Test tensor equality comparisons."""

    def test_eq_same_tensor(self):
        a = Tensor([1, 2, 3], dtype='float32')
        assert a == a

    def test_eq_same_values(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        b = Tensor([[1, 2], [3, 4]], dtype='float32')
        assert a == b

    def test_eq_different_values(self):
        a = Tensor([1, 2, 3], dtype='float32')
        b = Tensor([1, 2, 4], dtype='float32')
        assert not (a == b)

    def test_eq_different_shapes(self):
        a = Tensor([1, 2, 3], dtype='float32')
        b = Tensor([[1, 2, 3]], dtype='float32')
        assert not (a == b)

    def test_eq_different_dtypes(self):
        a = Tensor([1, 2, 3], dtype='float32')
        b = Tensor([1, 2, 3], dtype='float64')
        assert not (a == b)

    def test_eq_with_list(self):
        a = Tensor([1, 2, 3], dtype='float32')
        assert a == [1.0, 2.0, 3.0]

    def test_eq_invalid_type(self):
        a = Tensor([1, 2, 3], dtype='float32')
        with pytest.raises(ValueError):
            a == "string"


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

    def test_backward_non_scalar_with_grad(self):
        a = Tensor([1, 2, 3], dtype='float32', requires_grad=True)
        b = a * 2
        grad = Tensor([1, 1, 1], dtype='float32')
        b.backward(grad)
        assert a.grad is not None
        assert np.allclose(a.grad.tolist(), [2, 2, 2])

    def test_backward_add_with_broadcasting(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32', requires_grad=True)
        b = Tensor([1, 2], dtype='float32', requires_grad=True)
        c = a + b
        loss = c.sum()
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_backward_mul(self):
        a = Tensor([1, 2, 3], dtype='float32', requires_grad=True)
        b = Tensor([4, 5, 6], dtype='float32', requires_grad=True)
        c = a * b
        loss = c.sum()
        loss.backward()
        assert np.allclose(a.grad.tolist(), [4, 5, 6])
        assert np.allclose(b.grad.tolist(), [1, 2, 3])

    def test_backward_div(self):
        a = Tensor([4, 6], dtype='float32', requires_grad=True)
        b = Tensor([2, 3], dtype='float32', requires_grad=True)
        c = a / b
        loss = c.sum()
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_backward_matmul(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32', requires_grad=True)
        b = Tensor([[1, 0], [0, 1]], dtype='float32', requires_grad=True)
        c = a @ b
        loss = c.sum()
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_backward_chained_operations(self):
        a = Tensor([1, 2, 3], dtype='float32', requires_grad=True)
        b = a * 2
        c = b + 1
        d = c ** 2
        loss = d.sum()
        loss.backward()
        assert a.grad is not None

    def test_backward_sqrt(self):
        a = Tensor([4, 9, 16], dtype='float32', requires_grad=True)
        b = a.sqrt()
        loss = b.sum()
        loss.backward()
        assert a.grad is not None
        assert np.isclose(a.grad.tolist()[0], 0.25, rtol=1e-4)

    def test_backward_exp(self):
        # exp backward requires both input and output stored in _grad_fn
        # Verifies exp gradient tracking is set up correctly
        a = Tensor([0, 1], dtype='float32', requires_grad=True)
        b = a.exp()
        assert b.requires_grad
        assert b._grad_fn is not None
        assert b._grad_fn[0] == 'exp'

    def test_gradient_accumulates_on_multiple_backward(self):
        a = Tensor([1, 2, 3], dtype='float32', requires_grad=True)
        b = a * 2
        b.sum().backward()
        c = a * 3
        c.sum().backward()
        assert np.allclose(a.grad.tolist(), [5, 5, 5])

    def test_zero_grad_clears_gradient(self):
        a = Tensor([1, 2, 3], dtype='float32', requires_grad=True)
        b = a * 2
        c = b.sum()
        c.backward()
        assert a.grad is not None
        a.zero_grad()
        assert a.grad is None


class TestComplexComputationGraphs:
    """Test complex computation graphs with multiple branches."""

    def test_diamond_graph(self):
        a = Tensor([1, 2, 3], dtype='float32', requires_grad=True)
        b = a * 2
        c = a * 3
        d = b + c
        loss = d.sum()
        loss.backward()
        assert np.allclose(a.grad.tolist(), [5, 5, 5])

    def test_reused_tensor(self):
        a = Tensor([1, 2, 3], dtype='float32', requires_grad=True)
        b = a * a
        loss = b.sum()
        loss.backward()
        assert np.allclose(a.grad.tolist(), [2, 4, 6])


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


class TestBroadcastingEdgeCases:
    """Test broadcasting edge cases."""

    def test_broadcast_scalar_to_matrix(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        b = 5
        c = a + b
        assert np.allclose(c.tolist(), [[6, 7], [8, 9]])

    def test_broadcast_row_to_matrix(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        b = Tensor([10, 20], dtype='float32')
        c = a + b
        assert np.allclose(c.tolist(), [[11, 22], [13, 24]])

    def test_broadcast_column_to_matrix(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        b = Tensor([[10], [20]], dtype='float32')
        c = a + b
        assert np.allclose(c.tolist(), [[11, 12], [23, 24]])

    def test_broadcast_incompatible_shapes(self):
        a = Tensor([[1, 2, 3]], dtype='float32')
        b = Tensor([[1, 2]], dtype='float32')
        with pytest.raises(RuntimeError):
            a + b
