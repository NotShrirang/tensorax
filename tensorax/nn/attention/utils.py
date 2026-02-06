from tensorax.tensor import Tensor


def create_causal_mask(seq_len: int, batch_size: int = 1, num_heads: int = 1, device: str = 'cpu') -> Tensor:
    data = []
    for _ in range(batch_size * num_heads):
        for i in range(seq_len):
            for j in range(seq_len):
                data.append(0.0 if j <= i else float('-inf'))
    return Tensor(data, shape=(batch_size, num_heads, seq_len, seq_len), device=device)


def create_padding_mask(lengths: list, max_len: int, num_heads: int = 1, device: str = 'cpu') -> Tensor:
    batch_size = len(lengths)
    data = []
    for b in range(batch_size):
        for _ in range(num_heads):
            for _ in range(1):
                for j in range(max_len):
                    data.append(0.0 if j < lengths[b] else float('-inf'))
    return Tensor(data, shape=(batch_size, num_heads, 1, max_len), device=device)
