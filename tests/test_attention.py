import pytest
import math
import tensorax as ts
from tensorax import Tensor
import tensorax.functional as F


def reference_sdpa(q_list, k_list, v_list, mask_list, batch, heads, seq_q, seq_k, d_k, d_v):
    scale = 1.0 / math.sqrt(d_k)
    out = [0.0] * (batch * heads * seq_q * d_v)

    for b in range(batch):
        for h in range(heads):
            base = b * heads + h
            for i in range(seq_q):
                scores = []
                for j in range(seq_k):
                    s = 0.0
                    for d in range(d_k):
                        s += q_list[base * seq_q * d_k + i * d_k + d] * k_list[base * seq_k * d_k + j * d_k + d]
                    s *= scale
                    if mask_list is not None:
                        s += mask_list[base * seq_q * seq_k + i * seq_k + j]
                    scores.append(s)

                max_s = max(scores)
                exp_s = [math.exp(s - max_s) for s in scores]
                sum_e = sum(exp_s)
                attn = [e / sum_e for e in exp_s]

                for d in range(d_v):
                    val = 0.0
                    for j in range(seq_k):
                        val += attn[j] * v_list[base * seq_k * d_v + j * d_v + d]
                    out[base * seq_q * d_v + i * d_v + d] = val
    return out


def flatten_nested(data):
    if not isinstance(data, list):
        return [data]
    result = []
    for item in data:
        result.extend(flatten_nested(item))
    return result


def allclose(list_a, list_b, atol=1e-4):
    if len(list_a) != len(list_b):
        return False
    for a, b in zip(list_a, list_b):
        if abs(a - b) > atol:
            return False
    return True


class TestScaledDotProductAttention:

    def test_sdpa_shape_basic(self):
        batch_size, num_heads, seq_len, d_k, d_v = 2, 4, 8, 16, 16
        Q = Tensor.randn((batch_size, num_heads, seq_len, d_k))
        K = Tensor.randn((batch_size, num_heads, seq_len, d_k))
        V = Tensor.randn((batch_size, num_heads, seq_len, d_v))

        output = F.scaled_dot_product_attention(Q, K, V)

        assert output.shape == (batch_size, num_heads, seq_len, d_v)

    def test_sdpa_different_seq_lens(self):
        batch, heads, seq_q, seq_k, d_k, d_v = 1, 2, 4, 8, 8, 8
        Q = Tensor.randn((batch, heads, seq_q, d_k))
        K = Tensor.randn((batch, heads, seq_k, d_k))
        V = Tensor.randn((batch, heads, seq_k, d_v))

        output = F.scaled_dot_product_attention(Q, K, V)

        assert output.shape == (batch, heads, seq_q, d_v)

    def test_sdpa_with_mask(self):
        batch, heads, seq_len, d_k = 1, 1, 4, 4
        Q = Tensor.randn((batch, heads, seq_len, d_k))
        K = Tensor.randn((batch, heads, seq_len, d_k))
        V = Tensor.randn((batch, heads, seq_len, d_k))

        mask_data = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                row.append(0.0 if j <= i else -1e9)
            mask_data.append(row)
        mask = Tensor([[mask_data]])

        output = F.scaled_dot_product_attention(Q, K, V, mask=mask)
        assert output.shape == (batch, heads, seq_len, d_k)

        q_flat = flatten_nested(Q.tolist())
        k_flat = flatten_nested(K.tolist())
        v_flat = flatten_nested(V.tolist())
        m_flat = flatten_nested(mask.tolist())
        ref = reference_sdpa(q_flat, k_flat, v_flat, m_flat, batch, heads, seq_len, seq_len, d_k, d_k)
        out_flat = flatten_nested(output.tolist())
        assert allclose(out_flat, ref, atol=1e-3)

    def test_sdpa_values_single_query(self):
        Q = Tensor([[[[1.0, 0.0]]]])
        K = Tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        V = Tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

        output = F.scaled_dot_product_attention(Q, K, V)
        out_flat = flatten_nested(output.tolist())

        q_flat = [1.0, 0.0]
        k_flat = [1.0, 0.0, 0.0, 1.0]
        v_flat = [1.0, 0.0, 0.0, 1.0]
        ref = reference_sdpa(q_flat, k_flat, v_flat, None, 1, 1, 1, 2, 2, 2)

        assert allclose(out_flat, ref, atol=1e-4)

    def test_sdpa_attention_pattern(self):
        d_k = 4
        Q = Tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        K = Tensor([[[[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]]]])
        V = Tensor([[[[10.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]]]])

        output = F.scaled_dot_product_attention(Q, K, V)
        out_flat = flatten_nested(output.tolist())

        assert out_flat[0] > 3.0

    def test_sdpa_cpu_correctness(self):
        batch, heads, seq_q, seq_k, d_k, d_v = 2, 2, 4, 6, 8, 8
        Q = Tensor.randn((batch, heads, seq_q, d_k))
        K = Tensor.randn((batch, heads, seq_k, d_k))
        V = Tensor.randn((batch, heads, seq_k, d_v))

        output = F.scaled_dot_product_attention(Q, K, V)

        q_flat = flatten_nested(Q.tolist())
        k_flat = flatten_nested(K.tolist())
        v_flat = flatten_nested(V.tolist())
        ref = reference_sdpa(q_flat, k_flat, v_flat, None, batch, heads, seq_q, seq_k, d_k, d_v)
        out_flat = flatten_nested(output.tolist())

        assert allclose(out_flat, ref, atol=1e-4)

    def test_sdpa_batch_independence(self):
        heads, seq_len, d_k = 1, 4, 4
        Q = Tensor.randn((2, heads, seq_len, d_k))
        K = Tensor.randn((2, heads, seq_len, d_k))
        V = Tensor.randn((2, heads, seq_len, d_k))

        output = F.scaled_dot_product_attention(Q, K, V)
        out_data = output.tolist()

        assert out_data[0] != out_data[1]

    def test_sdpa_head_independence(self):
        batch, seq_len, d_k = 1, 4, 4
        Q = Tensor.randn((batch, 2, seq_len, d_k))
        K = Tensor.randn((batch, 2, seq_len, d_k))
        V = Tensor.randn((batch, 2, seq_len, d_k))

        output = F.scaled_dot_product_attention(Q, K, V)
        out_data = output.tolist()

        assert out_data[0][0] != out_data[0][1]

    def test_sdpa_invalid_shapes(self):
        with pytest.raises((ValueError, RuntimeError)):
            Q = Tensor.randn((4, 8))
            K = Tensor.randn((4, 8))
            V = Tensor.randn((4, 8))
            F.scaled_dot_product_attention(Q, K, V)

    def test_sdpa_mismatched_dk(self):
        with pytest.raises((ValueError, RuntimeError)):
            Q = Tensor.randn((1, 1, 4, 8))
            K = Tensor.randn((1, 1, 4, 16))
            V = Tensor.randn((1, 1, 4, 8))
            F.scaled_dot_product_attention(Q, K, V)

    def test_sdpa_numerical_stability(self):
        batch, heads, seq_len, d_k = 1, 1, 4, 4
        large_val = 100.0
        data = [large_val] * (batch * heads * seq_len * d_k)
        shape = (batch, heads, seq_len, d_k)
        Q = Tensor(data, shape=shape)
        K = Tensor(data, shape=shape)
        V = Tensor(data, shape=shape)

        output = F.scaled_dot_product_attention(Q, K, V)
        out_flat = flatten_nested(output.tolist())

        for val in out_flat:
            assert math.isfinite(val)


class TestSDPAReference:

    def test_against_manual_computation(self):
        Q = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        K = Tensor([[[[5.0, 6.0], [7.0, 8.0]]]])
        V = Tensor([[[[0.1, 0.2], [0.3, 0.4]]]])

        output = F.scaled_dot_product_attention(Q, K, V)

        q_flat = flatten_nested(Q.tolist())
        k_flat = flatten_nested(K.tolist())
        v_flat = flatten_nested(V.tolist())
        ref = reference_sdpa(q_flat, k_flat, v_flat, None, 1, 1, 2, 2, 2, 2)
        out_flat = flatten_nested(output.tolist())

        assert allclose(out_flat, ref, atol=1e-4)


def create_causal_mask(seq_len: int, device: str = 'cpu') -> Tensor:
    mask_data = []
    for b in range(1):
        heads = []
        for h in range(1):
            rows = []
            for i in range(seq_len):
                row = []
                for j in range(seq_len):
                    row.append(0.0 if j <= i else float('-inf'))
                rows.append(row)
            heads.append(rows)
        mask_data.append(heads)
    return Tensor(mask_data, device=device)


def create_padding_mask(lengths: list, max_len: int, device: str = 'cpu') -> Tensor:
    batch_size = len(lengths)
    mask_data = []
    for b in range(batch_size):
        heads = []
        for h in range(1):
            rows = []
            for i in range(1):
                row = []
                for j in range(max_len):
                    row.append(0.0 if j < lengths[b] else float('-inf'))
                rows.append(row)
            heads.append(rows)
        mask_data.append(heads)
    return Tensor(mask_data, device=device)


class TestMultiHeadAttentionLayer:

    def test_sdpa_layer_forward(self):
        from tensorax.nn.attention import ScaledDotProductAttention

        sdpa = ScaledDotProductAttention()
        Q = Tensor.randn((1, 2, 4, 8))
        K = Tensor.randn((1, 2, 4, 8))
        V = Tensor.randn((1, 2, 4, 8))

        output = sdpa(Q, K, V)
        assert output.shape == (1, 2, 4, 8)

    def test_sdpa_layer_with_mask(self):
        from tensorax.nn.attention import ScaledDotProductAttention

        sdpa = ScaledDotProductAttention()
        seq_len = 4
        Q = Tensor.randn((1, 1, seq_len, 8))
        K = Tensor.randn((1, 1, seq_len, 8))
        V = Tensor.randn((1, 1, seq_len, 8))
        mask = create_causal_mask(seq_len)

        output = sdpa(Q, K, V, mask=mask)
        assert output.shape == (1, 1, seq_len, 8)

    def test_sdpa_layer_repr(self):
        from tensorax.nn.attention import ScaledDotProductAttention

        sdpa = ScaledDotProductAttention()
        assert repr(sdpa) == "ScaledDotProductAttention()"


class TestAttentionUtilsAndVariants:

    def test_attention_utils_create_causal_mask(self):
        from tensorax.nn.attention import utils as attention_utils

        mask = attention_utils.create_causal_mask(seq_len=4, batch_size=2, num_heads=3)
        assert mask.shape == (2, 3, 4, 4)

        mask_data = mask.tolist()
        assert mask_data[0][0][0][0] == 0.0
        assert mask_data[0][0][0][1] == float('-inf')
        assert mask_data[1][2][3][0] == 0.0

    def test_attention_utils_create_padding_mask(self):
        from tensorax.nn.attention import utils as attention_utils

        mask = attention_utils.create_padding_mask(lengths=[1, 3], max_len=4, num_heads=2)
        assert mask.shape == (2, 2, 1, 4)

        mask_data = mask.tolist()
        # First sample length=1 -> only index 0 unmasked
        assert mask_data[0][0][0][0] == 0.0
        assert mask_data[0][0][0][1] == float('-inf')
        # Second sample length=3 -> indices [0,1,2] unmasked
        assert mask_data[1][1][0][2] == 0.0
        assert mask_data[1][1][0][3] == float('-inf')

    def test_sdpa_variants_output_shape(self):
        batch, heads, seq_len, d = 1, 1, 3, 4
        Q = Tensor.randn((batch, heads, seq_len, d))
        K = Tensor.randn((batch, heads, seq_len, d))
        V = Tensor.randn((batch, heads, seq_len, d))

        out_tiled = F.scaled_dot_product_attention_tiled(Q, K, V)
        out_flash = F.scaled_dot_product_attention_flash(Q, K, V)
        out_flash_opt = F.scaled_dot_product_attention_flash_optimized(Q, K, V)

        assert out_tiled.shape == (batch, heads, seq_len, d)
        assert out_flash.shape == (batch, heads, seq_len, d)
        assert out_flash_opt.shape == (batch, heads, seq_len, d)

    def test_sdpa_validates_tensor_device_consistency(self):
        Q = Tensor.randn((1, 1, 2, 2))
        K = Tensor.randn((1, 1, 2, 2))
        V = Tensor.randn((1, 1, 2, 2))

        # Force mismatch without requiring CUDA hardware.
        K.device = 'cuda'

        with pytest.raises(ValueError, match="same device"):
            F.scaled_dot_product_attention(Q, K, V)

    def test_sdpa_validates_mask_device_consistency(self):
        Q = Tensor.randn((1, 1, 2, 2))
        K = Tensor.randn((1, 1, 2, 2))
        V = Tensor.randn((1, 1, 2, 2))
        mask = Tensor.zeros((1, 1, 2, 2))

        # Force mismatch without requiring CUDA hardware.
        mask.device = 'cuda'

        with pytest.raises(ValueError, match="Mask must be on the same device"):
            F.scaled_dot_product_attention(Q, K, V, mask=mask)

    def test_sdpa_raises_when_backend_missing(self, monkeypatch):
        Q = Tensor.randn((1, 1, 2, 2))
        K = Tensor.randn((1, 1, 2, 2))
        V = Tensor.randn((1, 1, 2, 2))

        monkeypatch.setattr(F, "_C", None)

        with pytest.raises(RuntimeError, match=r"C\+\+ backend not available"):
            F.scaled_dot_product_attention(Q, K, V)

        with pytest.raises(RuntimeError, match=r"C\+\+ backend not available"):
            F._sdpa_variant(Q, K, V, None, lambda *_: None)

    def test_sdpa_variant_rejects_non_4d_inputs(self):
        Q = Tensor.randn((2, 2))
        K = Tensor.randn((2, 2))
        V = Tensor.randn((2, 2))

        with pytest.raises(ValueError, match="must be 4D tensors"):
            F._sdpa_variant(Q, K, V, None, lambda *_: None)

    def test_sdpa_variant_rejects_device_mismatch(self):
        Q = Tensor.randn((1, 1, 2, 2))
        K = Tensor.randn((1, 1, 2, 2))
        V = Tensor.randn((1, 1, 2, 2))
        V.device = 'cuda'

        with pytest.raises(ValueError, match="same device"):
            F._sdpa_variant(Q, K, V, None, lambda *_: None)

    def test_sdpa_variant_rejects_mask_device_mismatch(self):
        Q = Tensor.randn((1, 1, 2, 2))
        K = Tensor.randn((1, 1, 2, 2))
        V = Tensor.randn((1, 1, 2, 2))
        mask = Tensor.zeros((1, 1, 2, 2))
        mask.device = 'cuda'

        with pytest.raises(ValueError, match="Mask must be on the same device"):
            F._sdpa_variant(Q, K, V, mask, lambda *_: None)


class TestGroupedQueryAttention:
    """Test GQA layer."""
    def test_gqa_forward(self):
        # 4 query heads, 2 kv heads
        from tensorax.nn.attention.layers import GroupedQueryAttention
        gqa = GroupedQueryAttention(num_heads=4, num_kv_heads=2)
        
        # batch=1, seq_len=2, heads, head_dim=4
        q = Tensor.randn((1, 2, 4, 4))
        k = Tensor.randn((1, 2, 2, 4))
        v = Tensor.randn((1, 2, 2, 4))
        
        out = gqa(q, k, v)
        assert out.shape == (1, 2, 4, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
