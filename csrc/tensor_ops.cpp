#include "tensor_ops.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

namespace tensorax
{

    // TensorImpl implementation
    TensorImpl::TensorImpl(const std::vector<float> &input_data,
                           const std::vector<int64_t> &input_shape,
                           const std::string &input_dtype,
                           const std::string &input_device)
        : shape(input_shape), dtype(input_dtype), device(input_device)
    {

        size = 1;
        for (auto dim : shape)
        {
            size *= dim;
        }

        if (device == "cpu")
        {
            data = new float[size];
            std::copy(input_data.begin(), input_data.end(), data);
        }
        else if (device == "cuda")
        {
#ifdef WITH_CUDA
            data = static_cast<float *>(cuda_malloc(size * sizeof(float)));
            cuda_memcpy_h2d(data, input_data.data(), size * sizeof(float));
#else
            throw std::runtime_error("CUDA support not compiled");
#endif
        }
    }

    TensorImpl::~TensorImpl()
    {
        if (device == "cuda")
        {
#ifdef WITH_CUDA
            cuda_free(data);
#endif
        }
        else
        {
            delete[] data;
        }
    }

    std::vector<float> TensorImpl::to_vector() const
    {
        std::vector<float> result(size);
        if (device == "cpu")
        {
            std::copy(data, data + size, result.begin());
        }
        else
        {
#ifdef WITH_CUDA
            cuda_memcpy_d2h(result.data(), data, size * sizeof(float));
#endif
        }
        return result;
    }

    // Tensor creation
    TensorHandle create_tensor_cpu(const std::vector<float> &data,
                                   const std::vector<int64_t> &shape,
                                   const std::string &dtype)
    {
        return std::make_shared<TensorImpl>(data, shape, dtype, "cpu");
    }

    TensorHandle create_tensor_cuda(const std::vector<float> &data,
                                    const std::vector<int64_t> &shape,
                                    const std::string &dtype)
    {
        return std::make_shared<TensorImpl>(data, shape, dtype, "cuda");
    }

    TensorHandle copy_tensor(const TensorHandle &tensor)
    {
        auto data_vec = tensor->to_vector();
        return std::make_shared<TensorImpl>(data_vec, tensor->shape, tensor->dtype, tensor->device);
    }

    // Device transfer
    TensorHandle tensor_cpu_to_cuda(const TensorHandle &tensor)
    {
        if (tensor->device == "cuda")
            return tensor;
        auto data_vec = tensor->to_vector();
        return create_tensor_cuda(data_vec, tensor->shape, tensor->dtype);
    }

    TensorHandle tensor_cuda_to_cpu(const TensorHandle &tensor)
    {
        if (tensor->device == "cpu")
            return tensor;
        auto data_vec = tensor->to_vector();
        return create_tensor_cpu(data_vec, tensor->shape, tensor->dtype);
    }

    // Data access
    std::vector<float> tensor_to_list(const TensorHandle &tensor)
    {
        return tensor->to_vector();
    }

    // Element-wise operations
    TensorHandle add(const TensorHandle &a, const TensorHandle &b)
    {
        auto result = std::make_shared<TensorImpl>(std::vector<float>(a->size),
                                                   a->shape, a->dtype, a->device);
        if (a->device == "cpu")
        {
            add_cpu(a->data, b->data, result->data, a->size);
        }
        else
        {
#ifdef WITH_CUDA
            add_cuda(a->data, b->data, result->data, a->size);
#endif
        }
        return result;
    }

    TensorHandle broadcasting_add(const TensorHandle &a, const TensorHandle &b)
    {
        std::vector<int64_t> result_shape;
        int64_t ndim_a = a->shape.size();
        int64_t ndim_b = b->shape.size();
        int64_t ndim_result = std::max(ndim_a, ndim_b);

        for (int64_t i = 0; i < ndim_result; ++i)
        {
            int64_t dim_a = (i < ndim_result - ndim_a) ? 1 : a->shape[i - (ndim_result - ndim_a)];
            int64_t dim_b = (i < ndim_result - ndim_b) ? 1 : b->shape[i - (ndim_result - ndim_b)];
            if (dim_a != dim_b && dim_a != 1 && dim_b != 1)
            {
                throw std::runtime_error("Shapes are not broadcastable");
            }
            result_shape.push_back(std::max(dim_a, dim_b));
        }

        int64_t result_size = 1;
        for (auto dim : result_shape)
        {
            result_size *= dim;
        }

        auto result = std::make_shared<TensorImpl>(std::vector<float>(result_size), result_shape, a->dtype, a->device);
        if (a->device == "cpu")
        {
            broadcasting_add_cpu(a->data, b->data, result->data, a->shape, b->shape, result_shape);
        }
        else
        {
#ifdef WITH_CUDA
            broadcasting_add_cuda(a->data, b->data, result->data, a->shape, b->shape, result_shape);
#endif
        }
        return result;
    }

    TensorHandle subtract(const TensorHandle &a, const TensorHandle &b)
    {
        auto result = std::make_shared<TensorImpl>(std::vector<float>(a->size),
                                                   a->shape, a->dtype, a->device);
        if (a->device == "cpu")
        {
            sub_cpu(a->data, b->data, result->data, a->size);
        }
        else
        {
#ifdef WITH_CUDA
            sub_cuda(a->data, b->data, result->data, a->size);
#endif
        }
        return result;
    }

    TensorHandle multiply(const TensorHandle &a, const TensorHandle &b)
    {
        auto result = std::make_shared<TensorImpl>(std::vector<float>(a->size),
                                                   a->shape, a->dtype, a->device);
        if (a->device == "cpu")
        {
            mul_cpu(a->data, b->data, result->data, a->size);
        }
        else
        {
#ifdef WITH_CUDA
            mul_cuda(a->data, b->data, result->data, a->size);
#endif
        }
        return result;
    }

    TensorHandle divide(const TensorHandle &a, const TensorHandle &b)
    {
        auto result = std::make_shared<TensorImpl>(std::vector<float>(a->size),
                                                   a->shape, a->dtype, a->device);
        if (a->device == "cpu")
        {
            div_cpu(a->data, b->data, result->data, a->size);
        }
        else
        {
#ifdef WITH_CUDA
            div_cuda(a->data, b->data, result->data, a->size);
#endif
        }
        return result;
    }

    // Matrix operations
    TensorHandle matmul(const TensorHandle &a, const TensorHandle &b)
    {
        // Handle nD matrices: A: (..., m, k), B: (..., k, n) -> C: (..., m, n)
        int64_t m = a->shape[a->shape.size() - 2];
        int64_t k = a->shape[a->shape.size() - 1];
        int64_t n = b->shape[b->shape.size() - 1];

        // Calculate batch size (product of all dimensions except last two)
        int64_t batch_size = 1;
        for (size_t i = 0; i < a->shape.size() - 2; ++i)
        {
            batch_size *= a->shape[i];
        }

        // Result shape: same as a's shape but with last dimension replaced by n
        std::vector<int64_t> result_shape = a->shape;
        result_shape[result_shape.size() - 1] = n;

        // Calculate total result size
        int64_t result_size = batch_size * m * n;

        auto result = std::make_shared<TensorImpl>(std::vector<float>(result_size),
                                                   result_shape, a->dtype, a->device);
        if (a->device == "cpu")
        {
            matmul_cpu(a->data, b->data, result->data, batch_size, m, n, k);
        }
        else
        {
#ifdef WITH_CUDA
            matmul_cuda(a->data, b->data, result->data, batch_size, m, n, k);
#endif
        }
        return result;
    }

    TensorHandle matmul_tiled(const TensorHandle &a, const TensorHandle &b)
    {
        // Handle nD matrices: A: (..., m, k), B: (..., k, n) -> C: (..., m, n)
        int64_t m = a->shape[a->shape.size() - 2];
        int64_t k = a->shape[a->shape.size() - 1];
        int64_t n = b->shape[b->shape.size() - 1];

        // Calculate batch size (product of all dimensions except last two)
        int64_t batch_size = 1;
        for (size_t i = 0; i < a->shape.size() - 2; ++i)
        {
            batch_size *= a->shape[i];
        }

        // Result shape: same as a's shape but with last dimension replaced by n
        std::vector<int64_t> result_shape = a->shape;
        result_shape[result_shape.size() - 1] = n;

        // Calculate total result size
        int64_t result_size = batch_size * m * n;

        auto result = std::make_shared<TensorImpl>(std::vector<float>(result_size),
                                                   result_shape, a->dtype, a->device);
        if (a->device == "cpu")
        {
            matmul_cpu(a->data, b->data, result->data, batch_size, m, n, k);
        }
        else
        {
#ifdef WITH_CUDA
            matmul_tiled_cuda(a->data, b->data, result->data, batch_size, m, n, k);
#endif
        }
        return result;
    }

    TensorHandle matmul_with_shared_memory_coalescing(const TensorHandle &a, const TensorHandle &b, float alpha, float beta)
    {
        size_t a_dims = a->shape.size();
        size_t b_dims = b->shape.size();

        int64_t m = a->shape[a_dims - 2];
        int64_t k = a->shape[a_dims - 1];
        int64_t n = b->shape[b_dims - 1];

        int64_t batch_size = 1;
        for (size_t i = 0; i < a_dims - 2; ++i)
        {
            batch_size *= a->shape[i];
        }

        std::vector<int64_t> result_shape;
        for (size_t i = 0; i < a_dims - 2; ++i)
        {
            result_shape.push_back(a->shape[i]);
        }
        result_shape.push_back(m);
        result_shape.push_back(n);

        int64_t result_size = batch_size * m * n;
        auto result = std::make_shared<TensorImpl>(std::vector<float>(result_size), result_shape, a->dtype, a->device);

        if (a->device == "cuda")
        {
#ifdef WITH_CUDA
            matmul_cuda_shared_memory_coalesced_cuda(a->data, b->data, result->data, batch_size, m, n, k, alpha, beta);
#else
            throw std::runtime_error("CUDA support not compiled");
#endif
        }
        else
        {
            throw std::runtime_error("matmul_with_shared_memory_coalescing is only implemented for CUDA tensors");
        }
        return result;
    }

    TensorHandle matmul_with_shared_memory_cache_blocking(const TensorHandle &a, const TensorHandle &b, float alpha, float beta)
    {
        size_t a_dims = a->shape.size();
        size_t b_dims = b->shape.size();

        int64_t m = a->shape[a_dims - 2];
        int64_t k = a->shape[a_dims - 1];
        int64_t n = b->shape[b_dims - 1];

        int64_t batch_size = 1;
        for (size_t i = 0; i < a_dims - 2; ++i)
        {
            batch_size *= a->shape[i];
        }

        std::vector<int64_t> result_shape;
        for (size_t i = 0; i < a_dims - 2; ++i)
        {
            result_shape.push_back(a->shape[i]);
        }
        result_shape.push_back(m);
        result_shape.push_back(n);

        int64_t result_size = batch_size * m * n;
        auto result = std::make_shared<TensorImpl>(std::vector<float>(result_size), result_shape, a->dtype, a->device);

        if (a->device == "cuda")
        {
#ifdef WITH_CUDA
            matmul_cuda_shared_memory_cache_blocking_cuda(a->data, b->data, result->data, batch_size, m, n, k, alpha, beta);
#else
            throw std::runtime_error("CUDA support not compiled");
#endif
        }
        else
        {
            throw std::runtime_error("matmul_with_shared_memory_cache_blocking is only implemented for CUDA tensors");
        }
        return result;
    }

    TensorHandle transpose(const TensorHandle &a)
    {
        // Transpose last two dimensions for n-dimensional tensors
        auto result_shape = a->shape;
        std::swap(result_shape[result_shape.size() - 2], result_shape[result_shape.size() - 1]);

        auto result = std::make_shared<TensorImpl>(std::vector<float>(a->size), result_shape, a->dtype, a->device);

        int64_t rows = a->shape[a->shape.size() - 2];
        int64_t cols = a->shape[a->shape.size() - 1];

        // Calculate batch size (product of all dimensions except last two)
        int64_t batch_size = 1;
        for (size_t i = 0; i < a->shape.size() - 2; ++i)
        {
            batch_size *= a->shape[i];
        }

        if (a->device == "cpu")
        {
            int64_t matrix_size = rows * cols;

            // Transpose each matrix in the batch
            for (int64_t batch = 0; batch < batch_size; ++batch)
            {
                for (int64_t i = 0; i < rows; ++i)
                {
                    for (int64_t j = 0; j < cols; ++j)
                    {
                        result->data[batch * matrix_size + j * rows + i] = a->data[batch * matrix_size + i * cols + j];
                    }
                }
            }
        }
        else
        {
#ifdef WITH_CUDA
            transpose_cuda(a->data, result->data, batch_size, rows, cols);
#endif
        }

        return result;
    }

    TensorHandle sqrt_op(const TensorHandle &x)
    {
        auto result = std::make_shared<TensorImpl>(std::vector<float>(x->size),
                                                   x->shape, x->dtype, x->device);
        if (x->device == "cpu")
        {
            sqrt_cpu(x->data, result->data, x->size);
        }
        else
        {
#ifdef WITH_CUDA
            sqrt_cuda(x->data, result->data, x->size);
#endif
        }
        return result;
    }

    TensorHandle pow_op(const TensorHandle &x, float power)
    {
        auto result = std::make_shared<TensorImpl>(std::vector<float>(x->size),
                                                   x->shape, x->dtype, x->device);
        if (x->device == "cpu")
        {
            pow_cpu(x->data, result->data, power, x->size);
        }
        else
        {
#ifdef WITH_CUDA
            pow_cuda(x->data, result->data, power, x->size);
#endif
        }
        return result;
    }

    TensorHandle sum(const TensorHandle &x, int64_t dim)
    {
        // Simplified sum over a single dimension
        std::vector<int64_t> result_shape;
        int64_t result_size;

        if (dim == -1)
        {
            // Sum all elements - result is a scalar
            result_shape = {}; // Empty shape for scalar
            result_size = 1;
        }
        else
        {
            // Sum over specified dimension
            result_shape = x->shape;
            result_shape.erase(result_shape.begin() + dim);

            result_size = 1;
            for (auto dim_size : result_shape)
            {
                result_size *= dim_size;
            }
        }

        auto result = std::make_shared<TensorImpl>(std::vector<float>(result_size),
                                                   result_shape, x->dtype, x->device);

        if (x->device == "cpu")
        {
            sum_cpu(x->data, result->data, x->shape, dim);
        }
        else
        {
#ifdef WITH_CUDA
            sum_cuda(x->data, result->data, x->shape, dim);
#endif
        }
        return result;
    }

    TensorHandle mean(const TensorHandle &x, int64_t dim)
    {
        // Simplified mean over a single dimension
        std::vector<int64_t> result_shape;
        int64_t result_size;

        if (dim == -1)
        {
            // Mean of all elements - result is a scalar
            result_shape = {}; // Empty shape for scalar
            result_size = 1;
        }
        else
        {
            // Mean over specified dimension
            result_shape = x->shape;
            result_shape.erase(result_shape.begin() + dim);

            result_size = 1;
            for (auto dim_size : result_shape)
            {
                result_size *= dim_size;
            }
        }

        auto result = std::make_shared<TensorImpl>(std::vector<float>(result_size), result_shape, x->dtype, x->device);

        if (x->device == "cpu")
        {
            mean_cpu(x->data, result->data, x->shape, dim);
        }
        else
        {
#ifdef WITH_CUDA
            mean_cuda(x->data, result->data, x->shape, dim);
#endif
        }
        return result;
    }

    TensorHandle log(const TensorHandle &x)
    {
        auto result = std::make_shared<TensorImpl>(std::vector<float>(x->size),
                                                   x->shape, x->dtype, x->device);
        if (x->device == "cpu")
        {
            log_cpu(x->data, result->data, x->size);
        }
        else
        {
#ifdef WITH_CUDA
            log_cuda(x->data, result->data, x->size);
#endif
        }
        return result;
    }

    TensorHandle exp(const TensorHandle &x)
    {
        auto result = std::make_shared<TensorImpl>(std::vector<float>(x->size),
                                                   x->shape, x->dtype, x->device);
        if (x->device == "cpu")
        {
            exp_cpu(x->data, result->data, x->size);
        }
        else
        {
#ifdef WITH_CUDA
            exp_cuda(x->data, result->data, x->size);
#endif
        }
        return result;
    }

    // Activation functions
    TensorHandle relu(const TensorHandle &x)
    {
        auto result = std::make_shared<TensorImpl>(std::vector<float>(x->size),
                                                   x->shape, x->dtype, x->device);
        if (x->device == "cpu")
        {
            relu_cpu(x->data, result->data, x->size);
        }
        else
        {
#ifdef WITH_CUDA
            relu_cuda(x->data, result->data, x->size);
#endif
        }
        return result;
    }

    TensorHandle sigmoid(const TensorHandle &x)
    {
        auto result = std::make_shared<TensorImpl>(std::vector<float>(x->size),
                                                   x->shape, x->dtype, x->device);
        if (x->device == "cpu")
        {
            sigmoid_cpu(x->data, result->data, x->size);
        }
        else
        {
#ifdef WITH_CUDA
            sigmoid_cuda(x->data, result->data, x->size);
#endif
        }
        return result;
    }

    TensorHandle tanh_op(const TensorHandle &x)
    {
        auto result = std::make_shared<TensorImpl>(std::vector<float>(x->size),
                                                   x->shape, x->dtype, x->device);
        if (x->device == "cpu")
        {
            tanh_cpu(x->data, result->data, x->size);
        }
        else
        {
#ifdef WITH_CUDA
            tanh_cuda(x->data, result->data, x->size);
#endif
        }
        return result;
    }

    TensorHandle softmax(const TensorHandle &x, int64_t dim)
    {
        // Simplified softmax for last dimension
        auto result = std::make_shared<TensorImpl>(std::vector<float>(x->size), x->shape, x->dtype, x->device);

        // CPU implementation
        if (x->device == "cpu")
        {
            softmax_cpu(x->data, result->data, x->shape, dim);
        }

        return result;
    }

    // Loss functions
    TensorHandle mse_loss(const TensorHandle &pred, const TensorHandle &target)
    {
        auto result = std::make_shared<TensorImpl>(std::vector<float>(1),
                                                   std::vector<int64_t>{},
                                                   pred->dtype, pred->device);
        if (pred->device == "cpu")
        {
            mse_loss_cpu(pred->data, target->data, result->data[0], pred->size);
        }
        else
        {
#ifdef WITH_CUDA
            // CUDA implementation would go here
            throw std::runtime_error("CUDA MSE loss not implemented yet");
#else
            throw std::runtime_error("CUDA support not compiled");
#endif
        }
        return result;
    }

    TensorHandle cross_entropy_loss(const TensorHandle &pred, const TensorHandle &target)
    {
        // Simplified cross entropy: -mean(sum(target * log(pred)))
        std::vector<float> pred_data = pred->to_vector();
        std::vector<float> target_data = target->to_vector();

        if (pred_data.size() != target_data.size())
        {
            throw std::runtime_error("Pred and target must have the same size for cross entropy loss");
        }

        if (pred->device == "cpu")
        {
            TensorHandle result = std::make_shared<TensorImpl>(std::vector<float>(1),
                                                               std::vector<int64_t>{},
                                                               pred->dtype, pred->device);
            cross_entropy_loss_cpu(pred->data, target->data, result->data[0], pred->size);
            return result;
        }
        else if (pred->device == "cuda")
        {
#ifdef WITH_CUDA
            // CUDA implementation would go here
            throw std::runtime_error("CUDA cross entropy loss not implemented yet");
#else
            throw std::runtime_error("CUDA support not compiled");
#endif
        }
        else
        {
            throw std::runtime_error("Unsupported device for cross entropy loss");
        }
    }

    TensorHandle cross_entropy_from_logits(const TensorHandle &logits, const TensorHandle &targets, bool reduce_mean)
    {
        // logits: (batch_size, num_classes) or (num_classes,)
        // targets: (batch_size,) or scalar containing class indices

        if (logits->shape.size() < 1 || logits->shape.size() > 2)
        {
            throw std::runtime_error("Logits must be 1D (num_classes) or 2D (batch_size, num_classes)");
        }

        int64_t batch_size, num_classes;
        if (logits->shape.size() == 1)
        {
            // Single sample case
            batch_size = 1;
            num_classes = logits->shape[0];
        }
        else
        {
            // Batched case
            batch_size = logits->shape[0];
            num_classes = logits->shape[1];
        }

        if (targets->size != batch_size)
        {
            throw std::runtime_error("Targets size must match batch size");
        }

        if (logits->device == "cpu")
        {
            // Convert targets to int64_t indices
            std::vector<int64_t> target_indices(batch_size);
            for (int64_t i = 0; i < batch_size; ++i)
            {
                target_indices[i] = static_cast<int64_t>(targets->data[i]);
            }

            // Compute loss for each sample
            std::vector<float> losses(batch_size);
            cross_entropy_from_logits_cpu(logits->data, target_indices.data(),
                                          losses.data(), batch_size, num_classes);

            if (reduce_mean)
            {
                // Return mean loss
                float mean_loss = 0.0f;
                for (float loss : losses)
                {
                    mean_loss += loss;
                }
                mean_loss /= batch_size;

                return std::make_shared<TensorImpl>(std::vector<float>{mean_loss},
                                                    std::vector<int64_t>{},
                                                    logits->dtype, logits->device);
            }
            else
            {
                // Return individual losses
                return std::make_shared<TensorImpl>(losses,
                                                    std::vector<int64_t>{batch_size},
                                                    logits->dtype, logits->device);
            }
        }
        else if (logits->device == "cuda")
        {
#ifdef WITH_CUDA
            throw std::runtime_error("CUDA cross_entropy_from_logits not implemented yet");
#else
            throw std::runtime_error("CUDA support not compiled");
#endif
        }
        else
        {
            throw std::runtime_error("Unsupported device for cross_entropy_from_logits");
        }
    }

    // Random tensor
    TensorHandle randn(const std::vector<int64_t> &shape,
                       const std::string &dtype,
                       const std::string &device)
    {
        int64_t size = 1;
        for (auto dim : shape)
        {
            size *= dim;
        }

        std::vector<float> data(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (int64_t i = 0; i < size; ++i)
        {
            data[i] = dist(gen);
        }

        return std::make_shared<TensorImpl>(data, shape, dtype, device);
    }

} // namespace tensoraxx

// Python bindings
PYBIND11_MODULE(_C, m)
{
    m.doc() = "Tensorax C++ extension module - Pure implementation without NumPy";

    py::class_<tensorax::TensorImpl, std::shared_ptr<tensorax::TensorImpl>>(m, "TensorImpl")
        .def_readonly("shape", &tensorax::TensorImpl::shape)
        .def_readonly("size", &tensorax::TensorImpl::size)
        .def_readonly("dtype", &tensorax::TensorImpl::dtype)
        .def_readonly("device", &tensorax::TensorImpl::device);

    // Tensor creation
    m.def("create_tensor_cpu", &tensorax::create_tensor_cpu);
    m.def("create_tensor_cuda", &tensorax::create_tensor_cuda);
    m.def("copy_tensor", &tensorax::copy_tensor);

    // Device transfer
    m.def("tensor_cpu_to_cuda", &tensorax::tensor_cpu_to_cuda);
    m.def("tensor_cuda_to_cpu", &tensorax::tensor_cuda_to_cpu);

    // Data access
    m.def("tensor_to_list", &tensorax::tensor_to_list);

    // Operations
    m.def("add", &tensorax::add);
    m.def("broadcasting_add", &tensorax::broadcasting_add);
    m.def("subtract", &tensorax::subtract);
    m.def("multiply", &tensorax::multiply);
    m.def("divide", &tensorax::divide);
    m.def("transpose", &tensorax::transpose);
    m.def("sqrt", &tensorax::sqrt_op);
    m.def("pow", &tensorax::pow_op);
    m.def("sum", &tensorax::sum);
    m.def("mean", &tensorax::mean);
    m.def("log", &tensorax::log);
    m.def("exp", &tensorax::exp);

    // Activations
    m.def("relu", &tensorax::relu);
    m.def("sigmoid", &tensorax::sigmoid);
    m.def("tanh", &tensorax::tanh_op);
    m.def("softmax", &tensorax::softmax);

    // Losses
    m.def("mse_loss", &tensorax::mse_loss);
    m.def("cross_entropy_loss", &tensorax::cross_entropy_loss);
    m.def("cross_entropy_from_logits", &tensorax::cross_entropy_from_logits,
          py::arg("logits"), py::arg("targets"), py::arg("reduce_mean") = true);

    // Matmuls
    m.def("matmul", &tensorax::matmul);
    m.def("matmul_tiled", &tensorax::matmul_tiled);
    m.def("matmul_with_shared_memory_coalescing", &tensorax::matmul_with_shared_memory_coalescing,
          py::arg("a"), py::arg("b"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);
    m.def("matmul_with_shared_memory_cache_blocking", &tensorax::matmul_with_shared_memory_cache_blocking,
          py::arg("a"), py::arg("b"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);

    // Utility
    m.def("randn", &tensorax::randn);

#ifdef WITH_CUDA
    m.def("cuda_is_available", &tensorax::cuda_is_available);
#else
    m.def("cuda_is_available", []()
          { return false; });
#endif
}
