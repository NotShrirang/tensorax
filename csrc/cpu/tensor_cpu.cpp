#include "../tensor_ops.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace tensorax
{

    void add_cpu(const float *a, const float *b, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = a[i] + b[i];
        }
    }

    void broadcasting_add_cpu(const float *a, const float *b, float *out,
                              const std::vector<int64_t> &shape_a,
                              const std::vector<int64_t> &shape_b,
                              const std::vector<int64_t> &shape_out)
    {
        int64_t ndim_out = shape_out.size();
        int64_t ndim_a = shape_a.size();
        int64_t ndim_b = shape_b.size();

        std::vector<int64_t> stride_a_orig(ndim_a, 1);
        for (int64_t i = ndim_a - 2; i >= 0; --i)
        {
            stride_a_orig[i] = stride_a_orig[i + 1] * shape_a[i + 1];
        }

        std::vector<int64_t> stride_b_orig(ndim_b, 1);
        for (int64_t i = ndim_b - 2; i >= 0; --i)
        {
            stride_b_orig[i] = stride_b_orig[i + 1] * shape_b[i + 1];
        }

        std::vector<int64_t> stride_out(ndim_out, 1);
        for (int64_t i = ndim_out - 2; i >= 0; --i)
        {
            stride_out[i] = stride_out[i + 1] * shape_out[i + 1];
        }

        std::vector<int64_t> stride_a(ndim_out, 0);
        for (int64_t i = 0; i < ndim_out; ++i)
        {
            int64_t idx_a = i - (ndim_out - ndim_a);
            if (idx_a >= 0 && idx_a < ndim_a)
            {
                if (shape_a[idx_a] == shape_out[i])
                {
                    stride_a[i] = stride_a_orig[idx_a];
                }
                else if (shape_a[idx_a] == 1)
                {
                    stride_a[i] = 0;
                }
            }
        }

        std::vector<int64_t> stride_b(ndim_out, 0);
        for (int64_t i = 0; i < ndim_out; ++i)
        {
            int64_t idx_b = i - (ndim_out - ndim_b);
            if (idx_b >= 0 && idx_b < ndim_b)
            {
                if (shape_b[idx_b] == shape_out[i])
                {
                    stride_b[i] = stride_b_orig[idx_b];
                }
                else if (shape_b[idx_b] == 1)
                {
                    stride_b[i] = 0;
                }
            }
        }

        int64_t size_out = 1;
        for (auto dim : shape_out)
        {
            size_out *= dim;
        }

        for (int64_t idx = 0; idx < size_out; ++idx)
        {
            int64_t idx_a = 0;
            int64_t idx_b = 0;
            int64_t remaining = idx;

            for (int64_t i = 0; i < ndim_out; ++i)
            {
                int64_t coord = remaining / stride_out[i];
                remaining = remaining % stride_out[i];

                idx_a += coord * stride_a[i];
                idx_b += coord * stride_b[i];
            }

            out[idx] = a[idx_a] + b[idx_b];
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
    // A: (..., m, k), B: (..., k, n), C: (..., m, n)
    // Handles batched matrix multiplication for n-dimensional tensors
    void matmul_cpu(const float *a, const float *b, float *out,
                    int64_t batch_size, int64_t m, int64_t n, int64_t k)
    {
        int64_t matrix_size_a = m * k;
        int64_t matrix_size_b = k * n;
        int64_t matrix_size_out = m * n;

        for (int64_t batch = 0; batch < batch_size; ++batch)
        {
            const float *a_batch = a + batch * matrix_size_a;
            const float *b_batch = b + batch * matrix_size_b;
            float *out_batch = out + batch * matrix_size_out;

            for (int64_t i = 0; i < m; ++i)
            {
                for (int64_t j = 0; j < n; ++j)
                {
                    out_batch[i * n + j] = 0.0f;
                    for (int64_t p = 0; p < k; ++p)
                    {
                        out_batch[i * n + j] += a_batch[i * k + p] * b_batch[p * n + j];
                    }
                }
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

    void softmax_cpu(const float *in, float *out, const std::vector<int64_t> &shape, int64_t dim)
    {
        if (dim < 0)
        {
            dim = shape.size() + dim;
        }
        int64_t ndim = shape.size();
        if (ndim <= dim)
        {
            throw std::runtime_error("Dimension out of range for softmax");
        }
        const int64_t stride = shape[dim];
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i)
        {
            outer_size *= shape[i];
        }
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i)
        {
            inner_size *= shape[i];
        }
        for (int64_t outer = 0; outer < outer_size; ++outer)
        {
            for (int64_t inner = 0; inner < inner_size; ++inner)
            {
                // Calculate Max value for numerical stability
                float max_val = in[outer * stride * inner_size + inner];
                for (int64_t i = 0; i < stride; ++i)
                {
                    float val = in[outer * stride * inner_size + inner + inner_size * i];
                    if (val > max_val)
                    {
                        max_val = val;
                    }
                }

                // Compute exponentials and sum
                float sum = 0.0f;
                for (int64_t j = 0; j < stride; ++j)
                {
                    out[outer * stride * inner_size + inner + inner_size * j] =
                        std::exp(in[outer * stride * inner_size + inner + inner_size * j] - max_val);
                    sum += out[outer * stride * inner_size + inner + inner_size * j];
                }

                // Compute softmax output
                for (int64_t j = 0; j < stride; ++j)
                {
                    out[outer * stride * inner_size + inner + inner_size * j] =
                        out[outer * stride * inner_size + inner + inner_size * j] / sum;
                }
            }
        }
    }

    void sqrt_cpu(const float *in, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = std::sqrt(in[i]);
        }
    }

    void cross_entropy_loss_cpu(const float *pred, const float *target, float &loss, int64_t size)
    {
        float sum = 0.0f;
        for (int64_t i = 0; i < size; ++i)
        {
            float p = std::max(pred[i], 1e-12f); // prevent log(0)
            sum += target[i] * std::log(p);
        }
        loss = -sum;
    }

    // Cross entropy from logits for a single sample
    // More numerically stable than applying softmax then cross_entropy
    float cross_entropy_from_logits_single(const float *logits, int class_index, int64_t num_classes)
    {
        // Compute log-softmax using logsumexp trick for numerical stability
        float max_logit = logits[0];
        for (int64_t i = 1; i < num_classes; ++i)
            max_logit = std::max(max_logit, logits[i]);

        float sum = 0.0f;
        for (int64_t i = 0; i < num_classes; ++i)
            sum += std::exp(logits[i] - max_logit);

        float log_softmax = logits[class_index] - max_logit - std::log(sum);

        return -log_softmax;
    }

    // Cross entropy from logits for batched data
    // logits: (batch_size, num_classes)
    // targets: (batch_size,) containing class indices
    void cross_entropy_from_logits_cpu(const float *logits, const int64_t *targets,
                                       float *losses, int64_t batch_size, int64_t num_classes)
    {
        for (int64_t batch = 0; batch < batch_size; ++batch)
        {
            const float *batch_logits = logits + batch * num_classes;
            int64_t target_class = targets[batch];

            if (target_class < 0 || target_class >= num_classes)
            {
                throw std::runtime_error("Target class index out of bounds");
            }

            losses[batch] = cross_entropy_from_logits_single(batch_logits, target_class, num_classes);
        }
    }

    void mse_loss_cpu(const float *pred, const float *target, float &loss, int64_t size)
    {
        float sum = 0.0f;
        for (int64_t i = 0; i < size; ++i)
        {
            float diff = pred[i] - target[i];
            sum += diff * diff;
        }
        loss = sum / size;
    }

    void pow_cpu(const float *in, float *out, float power, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = std::pow(in[i], power);
        }
    }

    void sum_cpu(const float *in, float *out, const std::vector<int64_t> &shape, int64_t dim)
    {
        // Sum over the specified dimension
        if (dim == -1)
        {
            // Sum all elements
            float total = 0.0f;
            int64_t size = 1;
            for (auto s : shape)
            {
                size *= s;
            }
            for (int64_t i = 0; i < size; ++i)
            {
                total += in[i];
            }
            out[0] = total;
        }
        else
        {
            // Sum over specified dimension
            int64_t ndim = shape.size();

            // Calculate strides for input tensor
            std::vector<int64_t> strides(ndim, 1);
            for (int64_t i = ndim - 2; i >= 0; --i)
            {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Calculate output shape and strides
            std::vector<int64_t> out_shape;
            for (int64_t i = 0; i < ndim; ++i)
            {
                if (i != dim)
                {
                    out_shape.push_back(shape[i]);
                }
            }

            int64_t out_ndim = out_shape.size();
            std::vector<int64_t> out_strides(out_ndim > 0 ? out_ndim : 1, 1);
            for (int64_t i = out_ndim - 2; i >= 0; --i)
            {
                out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
            }

            int64_t out_size = 1;
            for (auto s : out_shape)
            {
                out_size *= s;
            }

            // Initialize output to zero
            for (int64_t i = 0; i < out_size; ++i)
            {
                out[i] = 0.0f;
            }

            // Calculate total input size
            int64_t total_size = 1;
            for (auto s : shape)
            {
                total_size *= s;
            }

            // Sum along the specified dimension
            for (int64_t idx = 0; idx < total_size; ++idx)
            {
                // Convert linear index to multi-dimensional coordinates
                std::vector<int64_t> coords(ndim);
                int64_t remaining = idx;
                for (int64_t i = 0; i < ndim; ++i)
                {
                    coords[i] = remaining / strides[i];
                    remaining = remaining % strides[i];
                }

                // Calculate output index by removing the summed dimension coordinate
                int64_t out_idx = 0;
                int64_t out_dim_idx = 0;
                for (int64_t i = 0; i < ndim; ++i)
                {
                    if (i != dim)
                    {
                        out_idx += coords[i] * out_strides[out_dim_idx];
                        out_dim_idx++;
                    }
                }

                out[out_idx] += in[idx];
            }
        }
    }

    void mean_cpu(const float *in, float *out, const std::vector<int64_t> &shape, int64_t dim)
    {
        // Compute mean by summing and dividing by the size of the dimension
        sum_cpu(in, out, shape, dim);

        int64_t dim_size;
        int64_t out_size;
        int64_t ndim = shape.size();

        if (dim == -1)
        {
            // Mean of all elements
            dim_size = 1;
            for (auto s : shape)
            {
                dim_size *= s;
            }
            out_size = 1;
        }
        else
        {
            // Mean over specified dimension
            dim_size = shape[dim];
            out_size = 1;
            for (int64_t i = 0; i < ndim; ++i)
            {
                if (i != dim)
                {
                    out_size *= shape[i];
                }
            }
        }

        for (int64_t i = 0; i < out_size; ++i)
        {
            out[i] /= dim_size;
        }
    }

    void log_cpu(const float *in, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = std::log(in[i]);
        }
    }

    void exp_cpu(const float *in, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = std::exp(in[i]);
        }
    }
} // namespace tensoraxx
