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