#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>

namespace tensorax
{
    class TensorImpl;

    using TensorHandle = std::shared_ptr<TensorImpl>;

    class TensorImpl
    {
    public:
        float *data;
        std::vector<int64_t> shape;
        int64_t size;
        std::string dtype;
        std::string device;

        TensorImpl(const std::vector<float> &data, const std::vector<int64_t> &shape,
                   const std::string &dtype, const std::string &device);
        ~TensorImpl();

        std::vector<float> to_vector() const;
    };

    TensorHandle create_tensor_cpu(const std::vector<float> &data,
                                   const std::vector<int64_t> &shape,
                                   const std::string &dtype);
    TensorHandle create_tensor_cuda(const std::vector<float> &data,
                                    const std::vector<int64_t> &shape,
                                    const std::string &dtype);
    TensorHandle copy_tensor(const TensorHandle &tensor);

    TensorHandle tensor_cpu_to_cuda(const TensorHandle &tensor);
    TensorHandle tensor_cuda_to_cpu(const TensorHandle &tensor);

    std::vector<float> tensor_to_list(const TensorHandle &tensor);

    TensorHandle add(const TensorHandle &a, const TensorHandle &b);
    TensorHandle broadcasting_add(const TensorHandle &a, const TensorHandle &b);
    TensorHandle subtract(const TensorHandle &a, const TensorHandle &b);
    TensorHandle multiply(const TensorHandle &a, const TensorHandle &b);
    TensorHandle divide(const TensorHandle &a, const TensorHandle &b);
    TensorHandle sqrt_op(const TensorHandle &x);
    TensorHandle pow_op(const TensorHandle &x, float power);

    TensorHandle matmul(const TensorHandle &a, const TensorHandle &b);
    TensorHandle transpose(const TensorHandle &a);
    TensorHandle sum(const TensorHandle &x, int64_t dim);
    TensorHandle mean(const TensorHandle &x, int64_t dim);

    TensorHandle relu(const TensorHandle &x);
    TensorHandle sigmoid(const TensorHandle &x);
    TensorHandle tanh_op(const TensorHandle &x);
    TensorHandle softmax(const TensorHandle &x, int64_t dim);

    TensorHandle mse_loss(const TensorHandle &pred, const TensorHandle &target);
    TensorHandle cross_entropy_loss(const TensorHandle &pred, const TensorHandle &target);
    TensorHandle cross_entropy_from_logits(const TensorHandle &logits, const TensorHandle &targets, bool reduce_mean = true);
    TensorHandle mse_loss(const TensorHandle &pred, const TensorHandle &target);

    TensorHandle randn(const std::vector<int64_t> &shape,
                       const std::string &dtype,
                       const std::string &device);

    bool cuda_is_available();

    void add_cpu(const float *a, const float *b, float *out, int64_t size);
    void broadcasting_add_cpu(const float *a, const float *b, float *out,
                              const std::vector<int64_t> &shape_a,
                              const std::vector<int64_t> &shape_b,
                              const std::vector<int64_t> &shape_out);
    void sub_cpu(const float *a, const float *b, float *out, int64_t size);
    void mul_cpu(const float *a, const float *b, float *out, int64_t size);
    void div_cpu(const float *a, const float *b, float *out, int64_t size);
    void matmul_cpu(const float *a, const float *b, float *out, int64_t batch_size, int64_t m, int64_t n, int64_t k);
    void relu_cpu(const float *in, float *out, int64_t size);
    void sigmoid_cpu(const float *in, float *out, int64_t size);
    void cross_entropy_loss_cpu(const float *pred, const float *target, float &loss, int64_t size);
    float cross_entropy_from_logits_single(const float *logits, int class_index, int64_t num_classes);
    void cross_entropy_from_logits_cpu(const float *logits, const int64_t *targets, float *losses, int64_t batch_size, int64_t num_classes);
    void mse_loss_cpu(const float *pred, const float *target, float &loss, int64_t size);
    void tanh_cpu(const float *in, float *out, int64_t size);
    void softmax_cpu(const float *in, float *out, const std::vector<int64_t> &shape, int64_t dim);
    void sqrt_cpu(const float *in, float *out, int64_t size);
    void pow_cpu(const float *in, float *out, float power, int64_t size);
    void sum_cpu(const float *in, float *out, const std::vector<int64_t> &shape, int64_t dim);
    void mean_cpu(const float *in, float *out, const std::vector<int64_t> &shape, int64_t dim);
    void log_cpu(const float *in, float *out, int64_t size);
    void exp_cpu(const float *in, float *out, int64_t size);

#ifdef WITH_CUDA
    void add_cuda(const float *a, const float *b, float *out, int64_t size);
    void broadcasting_add_cuda(const float *a, const float *b, float *out,
                               const std::vector<int64_t> &shape_a,
                               const std::vector<int64_t> &shape_b,
                               const std::vector<int64_t> &shape_out);
    void sub_cuda(const float *a, const float *b, float *out, int64_t size);
    void mul_cuda(const float *a, const float *b, float *out, int64_t size);
    void div_cuda(const float *a, const float *b, float *out, int64_t size);

    void relu_cuda(const float *in, float *out, int64_t size);
    void sigmoid_cuda(const float *in, float *out, int64_t size);
    void tanh_cuda(const float *in, float *out, int64_t size);
    void sqrt_cuda(const float *in, float *out, int64_t size);
    void pow_cuda(const float *in, float *out, float power, int64_t size);
    void sum_cuda(const float *in, float *out, const std::vector<int64_t> &shape, int64_t dim);
    void mean_cuda(const float *in, float *out, const std::vector<int64_t> &shape, int64_t dim);
    void log_cuda(const float *in, float *out, int64_t size);
    void exp_cuda(const float *in, float *out, int64_t size);

    void matmul_cuda(const float *a, const float *b, float *out, int64_t batch_size, int64_t m, int64_t n, int64_t k);
    void matmul_tiled_cuda(const float *a, const float *b, float *out, int64_t batch_size, int64_t m, int64_t n, int64_t k);
    void matmul_cuda_shared_memory_coalesced_cuda(const float *a, const float *b, float *c, int64_t batch_size, int64_t m, int64_t n, int64_t k, float alpha, float beta);
    void matmul_cuda_shared_memory_cache_blocking_cuda(const float *a, const float *b, float *c, int64_t batch_size, int64_t m, int64_t n, int64_t k, float alpha, float beta);

    void transpose_cuda(const float *in, float *out, int64_t batch_size, int64_t rows, int64_t cols);

    void *cuda_malloc(size_t size);
    void cuda_free(void *ptr);
    void cuda_memcpy_h2d(void *dst, const void *src, size_t size);
    void cuda_memcpy_d2h(void *dst, const void *src, size_t size);
#endif

}
