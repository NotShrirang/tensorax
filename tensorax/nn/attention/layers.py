import math
import random
from tensorax.tensor import Tensor
from tensorax import functional as F
from tensorax.nn.module import Module


class ScaledDotProductAttention:
    """Scaled Dot-Product Attention layer."""
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        return F.scaled_dot_product_attention(query, key, value, mask)

    def __call__(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        return self.forward(query, key, value, mask)
    
    def __repr__(self) -> str:
        return f"ScaledDotProductAttention()"

class GroupedQueryAttention:
    """Grouped Query Attention (GQA) layer."""
    def __init__(self, num_heads: int, num_kv_heads: int):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_queries_per_kv = num_heads // num_kv_heads

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        # query shape: (batch, seq_len, num_heads, head_dim)
        # key shape: (batch, seq_len, num_kv_heads, head_dim)
        # value shape: (batch, seq_len, num_kv_heads, head_dim)

        if self.num_queries_per_kv > 1:
            # Repeat KV heads for multiple query heads
            # Use repeat_interleave logic manually if repeat_interleave not explicitly available
            key_shapes = key.shape
            value_shapes = value.shape

            if hasattr(key, "repeat_interleave"):
                key = key.repeat_interleave(self.num_queries_per_kv, dim=2)
                value = value.repeat_interleave(self.num_queries_per_kv, dim=2)
            else:
                raise NotImplementedError("repeat_interleave required for GQA KV replication.")

        return F.scaled_dot_product_attention(query, key, value, mask)

    def __call__(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        return self.forward(query, key, value, mask)
    
    def __repr__(self) -> str:
        return f"GroupedQueryAttention(num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads})"


class MultiHeadAttention(Module):
    """Multi-Head Attention layer.

    Applies multi-head scaled dot-product attention as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. ``embed_dim`` must be
            divisible by ``num_heads``.
        bias: If ``True``, adds learnable bias to input/output projections.
        dropout: Dropout probability on attention weights (applied during
            training only). Default: ``0.0``.
    """

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Projection weights
        std = math.sqrt(2.0 / (embed_dim + embed_dim))
        for name, (rows, cols) in [
            ('q_weight', (embed_dim, embed_dim)),
            ('k_weight', (embed_dim, embed_dim)),
            ('v_weight', (embed_dim, embed_dim)),
            ('out_weight', (embed_dim, embed_dim)),
        ]:
            w_data = [random.gauss(0, std) for _ in range(rows * cols)]
            self._parameters[name] = Tensor(w_data, shape=(rows, cols), requires_grad=True)

        if bias:
            for name in ('q_bias', 'k_bias', 'v_bias', 'out_bias'):
                self._parameters[name] = Tensor.zeros((embed_dim,), requires_grad=True)
        else:
            for name in ('q_bias', 'k_bias', 'v_bias', 'out_bias'):
                self._parameters[name] = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            query:  (batch, seq_q, embed_dim)
            key:    (batch, seq_k, embed_dim)
            value:  (batch, seq_k, embed_dim)
            mask:   Optional (batch, num_heads, seq_q, seq_k) or broadcastable

        Returns:
            output: (batch, seq_q, embed_dim)
        """
        batch_size = query.shape[0]
        seq_q = query.shape[1]
        seq_k = key.shape[1]

        # Linear projections: (batch, seq, embed_dim) @ (embed_dim, embed_dim)^T
        q = F.linear(query, self._parameters['q_weight'], self._parameters['q_bias'])
        k = F.linear(key, self._parameters['k_weight'], self._parameters['k_bias'])
        v = F.linear(value, self._parameters['v_weight'], self._parameters['v_bias'])

        # Reshape to (batch, seq, num_heads, head_dim) then transpose to (batch, num_heads, seq, head_dim)
        q = q.reshape((batch_size, seq_q, self.num_heads, self.head_dim)).transpose(1, 2)
        k = k.reshape((batch_size, seq_k, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.reshape((batch_size, seq_k, self.num_heads, self.head_dim)).transpose(1, 2)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v, mask)

        # Concatenate heads: (batch, num_heads, seq_q, head_dim) -> (batch, seq_q, embed_dim)
        attn_output = attn_output.transpose(1, 2).reshape((batch_size, seq_q, self.embed_dim))

        # Output projection
        output = F.linear(attn_output, self._parameters['out_weight'], self._parameters['out_bias'])
        return output

    def __repr__(self) -> str:
        return (
            f"MultiHeadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, dropout={self.dropout})"
        )