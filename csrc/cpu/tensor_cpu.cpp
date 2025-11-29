#include "../tensor_ops.h"
#include <cmath>
#include <algorithm>

namespace tensora
{

    // Element-wise operations
    void add_cpu(const float *a, const float *b, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = a[i] + b[i];
        }
    }

    void broadcasting_add_cpu(const float *a, const float *b, float *out, int64_t size_a, int64_t size_b)
    {
        for (int64_t i = 0; i < size_b; ++i)
        {
            for (int64_t j = 0; j < size_a; ++j)
            {
                out[i * size_a + j] = a[j] + b[i];
            }
        }
    }

    void sub_cpu(const float *a, const float *b, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = a[i] - b[i];
        }
    }

    void mul_cpu(const float *a, const float *b, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = a[i] * b[i];
        }
    }

    void div_cpu(const float *a, const float *b, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = a[i] / b[i];
        }
    }

    // Matrix multiplication: C = A @ B
    // A: (m x k), B: (k x n), C: (m x n)
    void matmul_cpu(const float *a, const float *b, float *out,
                    int64_t m, int64_t n, int64_t k)
    {
        for (int64_t i = 0; i < m; ++i)
        {
            for (int64_t j = 0; j < n; ++j)
            {
                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p)
                {
                    sum += a[i * k + p] * b[p * n + j];
                }
                out[i * n + j] = sum;
            }
        }
    }

    // Activation functions
    void relu_cpu(const float *in, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = std::max(0.0f, in[i]);
        }
    }

    void sigmoid_cpu(const float *in, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = 1.0f / (1.0f + std::exp(-in[i]));
        }
    }

    void tanh_cpu(const float *in, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = std::tanh(in[i]);
        }
    }

    void sqrt_cpu(const float *in, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = std::sqrt(in[i]);
        }
    }

} // namespace tensora
