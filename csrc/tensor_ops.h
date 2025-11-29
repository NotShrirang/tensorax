#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>

namespace tensora
{

    // Forward declaration
    class TensorImpl;

    // Tensor handle for Python interface
    using TensorHandle = std::shared_ptr<TensorImpl>;

    // Tensor implementation class
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

    // Tensor creation functions
    TensorHandle create_tensor_cpu(const std::vector<float> &data,
                                   const std::vector<int64_t> &shape,
                                   const std::string &dtype);
    TensorHandle create_tensor_cuda(const std::vector<float> &data,
                                    const std::vector<int64_t> &shape,
                                    const std::string &dtype);
    TensorHandle copy_tensor(const TensorHandle &tensor);

    // Device transfer
    TensorHandle tensor_cpu_to_cuda(const TensorHandle &tensor);
    TensorHandle tensor_cuda_to_cpu(const TensorHandle &tensor);

    // Data access
    std::vector<float> tensor_to_list(const TensorHandle &tensor);

    // Element-wise operations
    TensorHandle add(const TensorHandle &a, const TensorHandle &b);
    TensorHandle broadcasting_add(const TensorHandle &a, const TensorHandle &b);
    TensorHandle subtract(const TensorHandle &a, const TensorHandle &b);
    TensorHandle multiply(const TensorHandle &a, const TensorHandle &b);
    TensorHandle divide(const TensorHandle &a, const TensorHandle &b);
    TensorHandle sqrt_op(const TensorHandle &x);

    // Matrix operations
    TensorHandle matmul(const TensorHandle &a, const TensorHandle &b);
    TensorHandle transpose(const TensorHandle &a);

    // Activation functions
    TensorHandle relu(const TensorHandle &x);
    TensorHandle sigmoid(const TensorHandle &x);
    TensorHandle tanh_op(const TensorHandle &x);
    TensorHandle softmax(const TensorHandle &x, int64_t dim);

    // Loss functions
    TensorHandle mse_loss(const TensorHandle &pred, const TensorHandle &target);
    TensorHandle cross_entropy_loss(const TensorHandle &pred, const TensorHandle &target);

    // Utility functions
    TensorHandle randn(const std::vector<int64_t> &shape,
                       const std::string &dtype,
                       const std::string &device);

    // CUDA availability
    bool cuda_is_available();

    // Low-level CPU operations
    void add_cpu(const float *a, const float *b, float *out, int64_t size);
    void broadcasting_add_cpu(const float *a, const float *b, float *out, int64_t size_a, int64_t size_b);
    void sub_cpu(const float *a, const float *b, float *out, int64_t size);
    void mul_cpu(const float *a, const float *b, float *out, int64_t size);
    void div_cpu(const float *a, const float *b, float *out, int64_t size);
    void matmul_cpu(const float *a, const float *b, float *out,
                    int64_t m, int64_t n, int64_t k);
    void relu_cpu(const float *in, float *out, int64_t size);
    void sigmoid_cpu(const float *in, float *out, int64_t size);
    void tanh_cpu(const float *in, float *out, int64_t size);
    void sqrt_cpu(const float *in, float *out, int64_t size);

#ifdef WITH_CUDA
    // Low-level CUDA operations
    void add_cuda(const float *a, const float *b, float *out, int64_t size);
    void broadcasting_add_cuda(const float *a, const float *b, float *out, int64_t size_a, int64_t size_b);
    void sub_cuda(const float *a, const float *b, float *out, int64_t size);
    void mul_cuda(const float *a, const float *b, float *out, int64_t size);
    void div_cuda(const float *a, const float *b, float *out, int64_t size);
    void matmul_cuda(const float *a, const float *b, float *out,
                     int64_t m, int64_t n, int64_t k);
    void relu_cuda(const float *in, float *out, int64_t size);
    void sigmoid_cuda(const float *in, float *out, int64_t size);
    void tanh_cuda(const float *in, float *out, int64_t size);
    void sqrt_cuda(const float *in, float *out, int64_t size);

    // CUDA utility functions
    void *cuda_malloc(size_t size);
    void cuda_free(void *ptr);
    void cuda_memcpy_h2d(void *dst, const void *src, size_t size);
    void cuda_memcpy_d2h(void *dst, const void *src, size_t size);
#endif

} // namespace tensora
