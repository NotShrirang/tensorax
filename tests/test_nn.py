import pytest
import numpy as np
from tensorax import nn, Tensor, cuda_is_available


class TestLinearLayer:
    """Test Linear (fully connected) layer."""

    def test_linear_initialization(self):
        layer = nn.Linear(10, 5)
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.weight.shape == (5, 10)
        assert layer.bias.shape == (5,)
        assert layer.weight.requires_grad
        assert layer.bias.requires_grad

    def test_linear_no_bias(self):
        layer = nn.Linear(10, 5, bias=False)
        assert layer.bias is None

    def test_linear_forward(self):
        layer = nn.Linear(3, 2)
        x = Tensor([[1, 2, 3]], dtype='float32')
        y = layer(x)
        assert y.shape == (1, 2)

    def test_linear_batch_forward(self):
        layer = nn.Linear(3, 2)
        x = Tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        y = layer(x)
        assert y.shape == (2, 2)

    def test_linear_backward(self):
        layer = nn.Linear(2, 1)
        x = Tensor([[1, 2]], dtype='float32', requires_grad=True)
        y = layer(x)
        y.backward(Tensor([[1]], dtype='float32'))
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None

    def test_linear_parameter_count(self):
        layer = nn.Linear(10, 5)
        params = list(layer.parameters())
        assert len(params) == 2  # weight and bias
        total_params = sum(p.size for p in params)
        assert total_params == 10 * 5 + 5  # weights + biases

    def test_linear_no_bias_parameter_count(self):
        layer = nn.Linear(10, 5, bias=False)
        params = list(layer.parameters())
        assert len(params) == 1  # only weight
        assert params[0].size == 50  # 10 * 5

    def test_linear_very_small_dimensions(self):
        linear = nn.Linear(1, 1)
        x = Tensor([[5]], dtype='float32')
        y = linear(x)
        assert y.shape == (1, 1)

    def test_linear_large_batch(self):
        linear = nn.Linear(10, 5)
        x = Tensor.randn((100, 10))
        y = linear(x)
        assert y.shape == (100, 5)


class TestActivationLayers:
    """Test activation function layers."""

    def test_relu_layer(self):
        layer = nn.ReLU()
        x = Tensor([[-1, 0, 1, 2]], dtype='float32')
        y = layer(x)
        expected = [[0, 0, 1, 2]]
        assert y.tolist() == expected

    def test_sigmoid_layer(self):
        layer = nn.Sigmoid()
        x = Tensor([[0]], dtype='float32')
        y = layer(x)
        assert abs(y.tolist()[0][0] - 0.5) < 1e-5

    def test_tanh_layer(self):
        layer = nn.Tanh()
        x = Tensor([[0]], dtype='float32')
        y = layer(x)
        assert abs(y.tolist()[0][0]) < 1e-5

    def test_softmax_layer_default_dim(self):
        layer = nn.Softmax()
        x = Tensor([[1, 2, 3]], dtype='float32')
        y = layer(x)
        values = y.tolist()[0]
        assert abs(sum(values) - 1.0) < 1e-5

    def test_softmax_layer_custom_dim(self):
        layer = nn.Softmax(dim=0)
        x = Tensor([[1, 2], [3, 4]], dtype='float32')
        y = layer(x)
        result = y.tolist()
        # Columns should sum to 1
        assert abs(result[0][0] + result[1][0] - 1.0) < 1e-5
        assert abs(result[0][1] + result[1][1] - 1.0) < 1e-5


class TestDropoutLayer:
    """Test dropout layer."""

    def test_dropout_training_mode(self):
        layer = nn.Dropout(p=0.5)
        layer.train()
        x = Tensor(np.ones((10, 10)), dtype='float32')
        y = layer(x)
        # In training mode, some values should be zeroed
        values = y.tolist()
        flattened = [v for row in values for v in row]
        zero_count = sum(1 for v in flattened if v == 0.0)
        nonzero_count = sum(1 for v in flattened if v != 0.0)
        assert zero_count > 0  # Some values should be dropped
        assert nonzero_count > 0  # Some values should remain

    def test_dropout_eval_mode(self):
        layer = nn.Dropout(p=0.5)
        layer.eval()
        x = Tensor(np.ones((10, 10)), dtype='float32')
        y = layer(x)
        # In eval mode, output should be same as input
        assert y.tolist() == x.tolist()

    def test_dropout_p_zero(self):
        layer = nn.Dropout(p=0.0)
        layer.train()
        x = Tensor([[1, 2, 3]], dtype='float32')
        y = layer(x)
        assert y.tolist() == x.tolist()

    def test_dropout_p_one(self):
        with pytest.raises(ValueError):
            layer = nn.Dropout(p=1.0)
    
    def test_dropout(self):
        layer = nn.Dropout(p=0.3)
        layer.train()
        x = Tensor([[1, 2, 3]], dtype='float32')
        y = layer(x)
        assert y.shape == x.shape

    def test_dropout_invalid_p(self):
        with pytest.raises(ValueError):
            nn.Dropout(p=1.5)
        with pytest.raises(ValueError):
            nn.Dropout(p=-0.1)

    def test_dropout_p_boundary(self):
        with pytest.raises(ValueError):
            nn.Dropout(p=1.0)

    def test_dropout_negative_p(self):
        with pytest.raises(ValueError):
            nn.Dropout(p=-0.1)


class TestSequential:
    """Test sequential container."""

    def test_sequential_creation(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        assert len(list(model.parameters())) == 4  # 2 weights + 2 biases

    def test_sequential_forward(self):
        model = nn.Sequential(
            nn.Linear(3, 2),
            nn.ReLU()
        )
        x = Tensor([[1, 2, 3]], dtype='float32')
        y = model(x)
        assert y.shape == (1, 2)

    def test_sequential_from_list(self):
        layers = [nn.Linear(3, 2), nn.ReLU()]
        model = nn.Sequential(layers)
        x = Tensor([[1, 2, 3]], dtype='float32')
        y = model(x)
        assert y.shape == (1, 2)

    def test_sequential_single_layer(self):
        model = nn.Sequential(nn.Linear(3, 2))
        x = Tensor([[1, 2, 3]], dtype='float32')
        y = model(x)
        assert y.shape == (1, 2)

    def test_sequential_empty(self):
        model = nn.Sequential()
        x = Tensor([[1, 2, 3]], dtype='float32')
        y = model(x)
        assert y.tolist() == x.tolist()

    def test_sequential_getitem_valid(self):
        seq = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        layer = seq[0]
        assert isinstance(layer, nn.Linear)

    def test_sequential_getitem_out_of_range(self):
        seq = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        with pytest.raises(IndexError):
            _ = seq[10]

    def test_sequential_empty_forward(self):
        seq = nn.Sequential()
        x = Tensor([[1, 2, 3]], dtype='float32')
        y = seq(x)
        assert y == x


class TestModuleBase:
    """Test base Module functionality."""

    def test_module_parameters(self):
        model = nn.Linear(10, 5)
        params = list(model.parameters())
        assert len(params) == 2

    def test_module_named_parameters(self):
        model = nn.Linear(10, 5)
        named_params = dict(model.named_parameters())
        assert 'weight' in named_params
        assert 'bias' in named_params
        assert named_params['weight'].shape == (5, 10)
        assert named_params['bias'].shape == (5,)

    def test_nested_module_parameters(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        params = list(model.parameters())
        assert len(params) == 4  # 2 per linear layer

    def test_nested_named_parameters(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        named_params = dict(model.named_parameters())
        # Should have names like '0.weight', '0.bias', '2.weight', '2.bias'
        assert '0.weight' in named_params
        assert '0.bias' in named_params
        assert '2.weight' in named_params
        assert '2.bias' in named_params

    def test_train_eval_modes(self):
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Dropout(0.5)
        )
        assert model.training
        for module in model._modules.values():
            assert module.training

        model.eval()
        assert not model.training
        for module in model._modules.values():
            assert not module.training

        model.train()
        assert model.training
        for module in model._modules.values():
            assert module.training

    def test_zero_grad(self):
        model = nn.Linear(10, 5)
        # Manually set some gradients
        model.weight.grad = Tensor.ones(model.weight.shape)
        model.bias.grad = Tensor.ones(model.bias.shape)

        model.zero_grad()
        assert model.weight.grad is None
        assert model.bias.grad is None

    @pytest.mark.skipif(not cuda_is_available(), reason="CUDA not available")
    def test_module_cuda_transfer(self):
        model = nn.Linear(10, 5)
        model.cuda()
        assert model.weight.device == 'cuda'
        assert model.bias.device == 'cuda'

    @pytest.mark.skipif(not cuda_is_available(), reason="CUDA not available")
    def test_module_cpu_transfer(self):
        model = nn.Linear(10, 5)
        model.cpu()
        assert model.weight.device == 'cpu'
        assert model.bias.device == 'cpu'

    def test_module_to_device(self):
        model = nn.Linear(10, 5)
        model.to('cpu')
        assert model.weight.device == 'cpu'

    def test_module_to_invalid_device(self):
        model = nn.Linear(10, 5)
        with pytest.raises(ValueError):
            model.to('invalid')

    def test_parameters_with_none_values(self):
        linear = nn.Linear(10, 5, bias=False)
        params = list(linear.parameters())
        assert len(params) == 1

    def test_nested_module_parameters_count(self):
        seq = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        params = list(seq.parameters())
        assert len(params) == 4


class TestModuleReprStr:
    """Test Module __repr__ and __str__ methods."""

    def test_module_repr(self):
        linear = nn.Linear(10, 5)
        repr_str = repr(linear)
        assert 'Linear' in repr_str
        assert 'training=' in repr_str

    def test_module_str(self):
        linear = nn.Linear(10, 5)
        str_repr = str(linear)
        assert 'Linear' in str_repr

    def test_sequential_repr(self):
        seq = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        repr_str = repr(seq)
        assert 'Sequential' in repr_str


class TestComplexNetworks:
    """Test complex network architectures."""

    def test_mlp_forward(self):
        # Multi-layer perceptron
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=-1)
        )

        x = Tensor(np.random.randn(32, 784), dtype='float32')
        y = model(x)
        assert y.shape == (32, 10)

        # Check softmax normalization
        for row in y.tolist():
            assert abs(sum(row) - 1.0) < 1e-5

    def test_mlp_backward(self):
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

        x = Tensor(np.random.randn(3, 10), dtype='float32', requires_grad=True)
        y = model(x)
        y.backward(Tensor(np.ones((3, 1)), dtype='float32'))

        assert x.grad is not None
        # Check that gradients flow through all layers
        for param in model.parameters():
            assert param.grad is not None

    def test_network_with_dropout(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

        x = Tensor(np.ones((5, 10)), dtype='float32')

        # Eval mode
        model.eval()
        y_eval1 = model(x)
        y_eval2 = model(x)

        # In eval mode, dropout should not affect output
        assert y_eval1.tolist() == y_eval2.tolist()

class TestEdgeCases:
    """Test edge cases for neural network layers."""

    def test_linear_single_feature(self):
        layer = nn.Linear(1, 1)
        x = Tensor([[5]], dtype='float32')
        y = layer(x)
        assert y.shape == (1, 1)

    def test_activations_on_empty_input(self):
        layer = nn.ReLU()
        x = Tensor([], dtype='float32')
        y = layer(x)
        assert y.shape == (0,)

    def test_sequential_with_non_module(self):
        # Should handle non-module objects gracefully
        with pytest.raises(AttributeError):
            model = nn.Sequential(nn.Linear(10, 5), "not a module")

    def test_module_without_parameters(self):
        layer = nn.ReLU()
        params = list(layer.parameters())
        assert len(params) == 0

    def test_deeply_nested_sequential(self):
        model = nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Linear(20, 5),
                nn.Softmax()
            )
        )
        x = Tensor(np.random.randn(2, 10), dtype='float32')
        y = model(x)
        assert y.shape == (2, 5)
        model = nn.Linear(10, 5)
        named_params = dict(model.named_parameters())
        assert 'weight' in named_params
        assert 'bias' in named_params
    
    def test_train_eval_mode(self):
        model = nn.Sequential(nn.Linear(10, 5), nn.Dropout(0.5))
        assert model.training
        model.eval()
        assert not model.training
        model.train()
        assert model.training


class TestTrainEvalModes:
    """Test train/eval mode switching."""

    def test_dropout_train_vs_eval(self):
        dropout = nn.Dropout(p=0.5)
        x = Tensor.ones((100,))
        
        dropout.train()
        y_train = dropout(x)
        
        dropout.eval()
        y_eval = dropout(x)
        
        assert np.allclose(y_eval.tolist(), x.tolist())

    def test_nested_module_train_eval(self):
        seq = nn.Sequential(nn.Linear(10, 5), nn.Dropout(0.5), nn.Linear(5, 2))
        seq.eval()
        assert seq.training is False
        for module in seq._modules.values():
            assert module.training is False

        seq.train()
        assert seq.training is True
