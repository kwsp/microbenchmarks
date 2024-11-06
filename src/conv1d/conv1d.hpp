#pragma once

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <Eigen/Dense>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <span>
#include <type_traits>
#include <vector>

/*
Conv1d with BLAS using the im2col method + gemm
"valid" mode

im2col_matrix must be preallocated to have size output_size * kernel_size,
where output_size = input_size - kernel_size + 1
*/
template <typename T>
void conv1d_openblas(std::span<const T> input, std::span<const T> kernel,
                     std::span<T> im2col_matrix, std::span<T> output)
  requires(std::is_floating_point_v<T>)
{
  const int input_size = input.size();
  const int kernel_size = kernel.size();
  const int output_size = input_size - kernel_size + 1;

  // Allocate memory for the im2col matrix
  // Perform im2col transformation
  for (int i = 0; i < output_size; ++i) {
    for (int j = 0; j < kernel_size; ++j) {
      im2col_matrix[i * kernel_size + j] = input[i + j];
    }
  }

  // Perform matrix multiplication: output = im2col_matrix * kernel
  // cblas_sgemm performs: C = alpha * A * B + beta * C
  // Here, we set alpha=1 and beta=0
  if constexpr (std::is_same_v<T, float>) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, output_size, 1,
                kernel_size, 1.0F, im2col_matrix.data(), kernel_size,
                kernel.data(), 1, 0.0F, output.data(), 1);
  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, output_size, 1,
                kernel_size, 1.0, im2col_matrix.data(), kernel_size,
                kernel.data(), 1, 0.0, output.data(), 1);
  } else {
    static_assert(false, "Unsupported type");
  }
}

/*
Conv1d with BLAS using the im2col method + gemm.
"same" mode

im2col_matrix must be preallocated to have size output_size * kernel_size,
where output_size = input_size - kernel_size + 1
*/
template <typename T>
void conv1d_openblas_same(std::span<const T> input, std::span<const T> kernel,
                          std::span<T> im2col_matrix, std::span<T> output)
  requires(std::is_floating_point_v<T>)
{
  const int input_size = input.size();
  const int kernel_size = kernel.size();

  // Calculate the required padding for "same" mode
  const int padding = (kernel_size - 1) / 2;
  const int padded_size = input_size + 2 * padding;

  // Create a padded input
  std::vector<T> padded_input(padded_size, 0.0F);
  for (int i = 0; i < input_size; ++i) {
    padded_input[i + padding] = input[i];
  }

  // The output size will be the same as the original input size
  const int output_size = input_size;

  // Perform im2col transformation on the padded input
  for (int i = 0; i < output_size; ++i) {
    for (int j = 0; j < kernel_size; ++j) {
      im2col_matrix[i * kernel_size + j] = padded_input[i + j];
    }
  }

  if constexpr (std::is_same_v<T, float>) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, output_size, 1,
                kernel_size, 1.0F, im2col_matrix.data(), kernel_size,
                kernel.data(), 1, 0.0F, output.data(), 1);
  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, output_size, 1,
                kernel_size, 1.0, im2col_matrix.data(), kernel_size,
                kernel.data(), 1, 0.0, output.data(), 1);
  } else {
    static_assert(false, "Unsupported type");
  }

  return output;
}

#ifdef __APPLE__

/*
Conv1d using Accelerate's vDSP

'valid' mode. output size must be (input.size() - kernel.size() + 1)
*/
template <typename T>
void conv1d_vdsp(const std::span<const T> input,
                 const std::span<const T> kernel, std::span<T> output)
  requires(std::is_floating_point_v<T>)
{
  const int input_size = input.size();
  const int kernel_size = kernel.size();
  const int output_size = input_size - kernel_size + 1;

  if constexpr (std::is_same_v<T, float>) {
    vDSP_conv(input.data(), 1, kernel.data(), 1, output.data(), 1, output_size,
              kernel_size);
  } else if constexpr (std::is_same_v<T, double>) {
    vDSP_convD(input.data(), 1, kernel.data(), 1, output.data(), 1, output_size,
               kernel_size);
  }
}
#endif

/*
Eigen
*/
template <typename T>
void conv1d_eigen(const std::span<const T> input_,
                  const std::span<const T> kernel_, std::span<T> output_) {
  Eigen::Map<const Eigen::VectorX<T>> input(input_.data(), input_.size());
  Eigen::Map<const Eigen::VectorX<T>> kernel(kernel_.data(), kernel_.size());
  Eigen::Map<Eigen::VectorX<T>> output(output_.data(), output_.size());

  auto p = input.data();

  const int input_size = input.size();
  const int kernel_size = kernel.size();
  const int output_size = input_size - kernel_size + 1;

  // Perform the 1D convolution
  for (int i = 0; i < output_size; ++i) {
    output(i) = input.segment(i, kernel_size).dot(kernel);
  }
}
