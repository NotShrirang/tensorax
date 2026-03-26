from tensorax.tensor import Tensor
from tensorax import functional as F


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