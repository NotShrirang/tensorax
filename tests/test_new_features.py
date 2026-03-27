"""Tests for Embedding, GELU, SiLU, MultiHeadAttention, and LR schedulers."""

import math
import pytest
from tensorax import Tensor, nn, optim, functional as F, lr_scheduler


# ── GELU / SiLU activations ────────────────────────────────────────────────

class TestGELU:
    def test_gelu_functional_basic(self):
        x = Tensor([0.0, 1.0, -1.0], shape=(3,))
        y = F.gelu(x)
        vals = y.tolist()
        # GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
        assert abs(vals[0] - 0.0) < 0.01
        assert abs(vals[1] - 0.841) < 0.02
        assert abs(vals[2] - (-0.159)) < 0.02

    def test_gelu_functional_2d(self):
        x = Tensor([0.0, 1.0, -1.0, 2.0], shape=(2, 2))
        y = F.gelu(x)
        assert y.shape == (2, 2)

    def test_gelu_layer(self):
        layer = nn.GELU()
        x = Tensor([0.5, -0.5], shape=(2,))
        y = layer(x)
        assert y.shape == (2,)

    def test_gelu_backward(self):
        x = Tensor([1.0, -1.0], shape=(2,), requires_grad=True)
        y = F.gelu(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2,)


class TestSiLU:
    def test_silu_functional_basic(self):
        x = Tensor([0.0, 1.0, -1.0], shape=(3,))
        y = F.silu(x)
        vals = y.tolist()
        # SiLU(0) = 0, SiLU(1) = 1*sigmoid(1) ≈ 0.731, SiLU(-1) = -1*sigmoid(-1) ≈ -0.269
        assert abs(vals[0] - 0.0) < 0.01
        assert abs(vals[1] - 0.731) < 0.02
        assert abs(vals[2] - (-0.269)) < 0.02

    def test_silu_functional_2d(self):
        x = Tensor([0.0, 1.0, -1.0, 2.0], shape=(2, 2))
        y = F.silu(x)
        assert y.shape == (2, 2)

    def test_silu_layer(self):
        layer = nn.SiLU()
        x = Tensor([0.5, -0.5], shape=(2,))
        y = layer(x)
        assert y.shape == (2,)

    def test_silu_backward(self):
        x = Tensor([1.0, -1.0], shape=(2,), requires_grad=True)
        y = F.silu(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2,)


# ── Embedding ───────────────────────────────────────────────────────────────

class TestEmbedding:
    def test_basic_lookup(self):
        emb = nn.Embedding(10, 4)
        indices = Tensor([0, 2, 5], shape=(3,))
        out = emb(indices)
        assert out.shape == (3, 4)

    def test_2d_indices(self):
        emb = nn.Embedding(10, 4)
        indices = Tensor([0, 1, 2, 3], shape=(2, 2))
        out = emb(indices)
        assert out.shape == (2, 2, 4)

    def test_embedding_values_match_weight(self):
        emb = nn.Embedding(5, 3)
        weight = emb.weight.tolist()
        indices = Tensor([0, 3], shape=(2,))
        out = emb(indices)
        out_list = out.tolist()
        for j in range(3):
            assert abs(out_list[0][j] - weight[0][j]) < 1e-6
            assert abs(out_list[1][j] - weight[3][j]) < 1e-6

    def test_embedding_out_of_range(self):
        emb = nn.Embedding(5, 3)
        indices = Tensor([10], shape=(1,))
        with pytest.raises(IndexError):
            emb(indices)

    def test_embedding_parameter_count(self):
        emb = nn.Embedding(100, 32)
        params = list(emb.parameters())
        assert len(params) == 1
        assert params[0].shape == (100, 32)

    def test_embedding_backward(self):
        emb = nn.Embedding(5, 3)
        indices = Tensor([1, 3], shape=(2,))
        out = emb(indices)
        loss = out.sum()
        loss.backward()
        assert emb.weight.grad is not None
        assert emb.weight.grad.shape == (5, 3)

    def test_embedding_in_sequential(self):
        """Embedding can be part of a module hierarchy."""
        emb = nn.Embedding(10, 8)
        assert isinstance(emb, nn.Module)
        params = list(emb.parameters())
        assert len(params) == 1


# ── MultiHeadAttention ──────────────────────────────────────────────────────

class TestMultiHeadAttention:
    def test_creation(self):
        mha = nn.MultiHeadAttention(embed_dim=16, num_heads=4)
        assert mha.embed_dim == 16
        assert mha.num_heads == 4
        assert mha.head_dim == 4

    def test_invalid_head_count(self):
        with pytest.raises(ValueError):
            nn.MultiHeadAttention(embed_dim=15, num_heads=4)

    def test_parameter_count_with_bias(self):
        mha = nn.MultiHeadAttention(embed_dim=8, num_heads=2, bias=True)
        params = list(mha.parameters())
        # 4 weight matrices (8x8) + 4 biases (8) = 4 + 4 = 8 params
        assert len(params) == 8

    def test_parameter_count_no_bias(self):
        mha = nn.MultiHeadAttention(embed_dim=8, num_heads=2, bias=False)
        params = list(mha.parameters())
        # 4 weight matrices only
        assert len(params) == 4

    def test_repr(self):
        mha = nn.MultiHeadAttention(embed_dim=32, num_heads=4)
        r = repr(mha)
        assert "MultiHeadAttention" in r
        assert "embed_dim=32" in r
        assert "num_heads=4" in r


# ── LR Schedulers ───────────────────────────────────────────────────────────

class TestStepLR:
    def _make_opt(self, lr=0.1):
        param = Tensor([1.0], requires_grad=True)
        return optim.SGD([param], lr=lr)

    def test_basic_decay(self):
        opt = self._make_opt(0.1)
        sched = lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)
        # epoch 0: lr = 0.1
        assert abs(opt.lr - 0.1) < 1e-9
        sched.step()  # epoch 1
        assert abs(opt.lr - 0.1) < 1e-9
        sched.step()  # epoch 2
        assert abs(opt.lr - 0.1) < 1e-9
        sched.step()  # epoch 3 -> decay
        assert abs(opt.lr - 0.01) < 1e-9

    def test_multiple_decays(self):
        opt = self._make_opt(0.1)
        sched = lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        for _ in range(4):
            sched.step()
        # epoch 4: 0.1 * 0.5^2 = 0.025
        assert abs(opt.lr - 0.025) < 1e-9

    def test_get_last_lr(self):
        opt = self._make_opt(0.1)
        sched = lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        assert abs(sched.get_last_lr() - 0.1) < 1e-9


class TestExponentialLR:
    def test_basic_decay(self):
        param = Tensor([1.0], requires_grad=True)
        opt = optim.SGD([param], lr=0.1)
        sched = lr_scheduler.ExponentialLR(opt, gamma=0.9)
        assert abs(opt.lr - 0.1) < 1e-9
        sched.step()  # epoch 1
        assert abs(opt.lr - 0.09) < 1e-9
        sched.step()  # epoch 2
        assert abs(opt.lr - 0.081) < 1e-9


class TestCosineAnnealingLR:
    def test_reaches_min(self):
        param = Tensor([1.0], requires_grad=True)
        opt = optim.SGD([param], lr=0.1)
        sched = lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
        for _ in range(10):
            sched.step()
        # At T_max, lr should be eta_min
        assert abs(opt.lr - 0.0) < 1e-6

    def test_cosine_shape(self):
        param = Tensor([1.0], requires_grad=True)
        opt = optim.SGD([param], lr=1.0)
        sched = lr_scheduler.CosineAnnealingLR(opt, T_max=4, eta_min=0.0)
        lrs = [opt.lr]
        for _ in range(4):
            sched.step()
            lrs.append(opt.lr)
        # LRs should decrease monotonically from 1.0 to 0.0
        assert lrs[0] > lrs[2] > lrs[4]


class TestLinearLR:
    def test_warmup(self):
        param = Tensor([1.0], requires_grad=True)
        opt = optim.SGD([param], lr=0.1)
        sched = lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=10)
        # epoch 0: factor = 0.1, lr = 0.01
        assert abs(opt.lr - 0.01) < 1e-9
        for _ in range(10):
            sched.step()
        # epoch 10: factor = 1.0, lr = 0.1
        assert abs(opt.lr - 0.1) < 1e-9

    def test_past_total_iters(self):
        param = Tensor([1.0], requires_grad=True)
        opt = optim.SGD([param], lr=0.1)
        sched = lr_scheduler.LinearLR(opt, start_factor=0.5, end_factor=1.0, total_iters=5)
        for _ in range(10):
            sched.step()
        # Should clamp at end_factor
        assert abs(opt.lr - 0.1) < 1e-9


class TestMultiStepLR:
    def test_milestones(self):
        param = Tensor([1.0], requires_grad=True)
        opt = optim.SGD([param], lr=0.1)
        sched = lr_scheduler.MultiStepLR(opt, milestones=[3, 7], gamma=0.1)
        # epoch 0: 0.1
        assert abs(opt.lr - 0.1) < 1e-9
        for _ in range(3):
            sched.step()
        # epoch 3: 0.01
        assert abs(opt.lr - 0.01) < 1e-9
        for _ in range(4):
            sched.step()
        # epoch 7: 0.001
        assert abs(opt.lr - 0.001) < 1e-9


class TestSchedulerValidation:
    def test_invalid_optimizer(self):
        with pytest.raises(TypeError):
            lr_scheduler.StepLR("not_an_optimizer", step_size=5)
