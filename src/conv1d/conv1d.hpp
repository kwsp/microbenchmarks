#pragma once

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <Eigen/Dense>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <kfr/all.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/opencv.hpp>
#include <span>
#include <type_traits>
#include <vector>

enum class ConvMode { Full, Same, Valid };

/*
Naive
*/
template <typename T, ConvMode Mode = ConvMode::Full>
void conv1d_naive(const std::span<const T> input,
                  const std::span<const T> kernel, std::span<T> output) {
  int output_size = 0;
  int pad = 0;

  if constexpr (Mode == ConvMode::Full) {
    output_size = input.size() + kernel.size() - 1;
    pad = kernel.size() - 1;
  } else if constexpr (Mode == ConvMode::Same) {
    output_size = input.size();
    pad = (kernel.size() - 1) / 2;
  } else if constexpr (Mode == ConvMode::Valid) {
    output_size = input.size() - kernel.size() + 1;
  }

  // Resize output to match output_size
  if (output.size() < output_size) {
    throw std::invalid_argument(
        "Output span size is too small for the selected mode");
  }

  for (int i = 0; i < output_size; ++i) {
    output[i] = 0; // Initialize to zero
    for (int j = 0; j < kernel.size(); ++j) {
      int input_index;

      // For "Valid" mode, adjust input index calculation to ensure no padding
      // and full overlap only
      if constexpr (Mode == ConvMode::Valid) {
        input_index = i + j; // Only include fully overlapping elements
      } else {
        input_index = i + j - pad;
      }

      if (input_index >= 0 && input_index < input.size()) {
        output[i] += input[input_index] * kernel[j];
      }
    }
  }
}

/*
Conv1d with BLAS using the im2col method + gemm
"valid" mode

im2col_matrix must be preallocated to have size output_size * kernel_size,
where output_size = input_size - kernel_size + 1
*/
template <typename T>
void conv1d_BLAS_im2col(std::span<const T> input, std::span<const T> kernel,
                        std::span<T> im2col_matrix, std::span<T> output)
  requires(std::is_floating_point_v<T>)
{
  const int input_size = input.size();
  const int kernel_size = kernel.size();
  const int output_size = input_size - kernel_size + 1;

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
void conv1d_BLAS_same(std::span<const T> input, std::span<const T> kernel,
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
void conv1d_vDSP(const std::span<const T> input,
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
Eigen (valid mode)
*/
template <typename T>
void conv1d_eigen(const std::span<const T> input_,
                  const std::span<const T> kernel_, std::span<T> output_) {
  Eigen::Map<const Eigen::VectorX<T>> input(input_.data(), input_.size());
  Eigen::Map<const Eigen::VectorX<T>> kernel(kernel_.data(), kernel_.size());
  Eigen::Map<Eigen::VectorX<T>> output(output_.data(), output_.size());

  const int input_size = input.size();
  const int kernel_size = kernel.size();
  const int output_size = input_size - kernel_size + 1;

  // Perform the 1D convolution
  for (int i = 0; i < output_size; ++i) {
    output(i) = input.segment(i, kernel_size).dot(kernel);
  }
}

/*
KFR (same mode)
*/
template <typename T>
void conv1d_KFR_fir(const std::span<const T> input,
                    const std::span<const T> kernel, std::span<T> output) {

  auto input_ = kfr::make_univector(input.data(), input.size());
  auto kernel_ = kfr::make_univector(kernel.data(), kernel.size());
  auto output_ = kfr::make_univector(output.data(), output.size());

  kfr::filter_fir<T> filter(kernel_);
  filter.apply(output_, input_);
}

#ifndef __APPLE__
template <typename T>
void conv1d_kfr_oa(const std::span<const T> input,
                   const std::span<const T> kernel, std::span<T> output) {

  auto input_ = kfr::make_univector(input.data(), input.size());
  auto kernel_ = kfr::make_univector(kernel.data(), kernel.size());
  auto output_ = kfr::make_univector(output.data(), output.size());

  kfr::convolve_filter<T> filter(kernel_);
  filter.apply(output_, input_);
}
#endif

/*
OpenCV (same mode)
*/

template <typename T> auto spanToMat1D(const std::span<const T> &span) {
  // Create a Mat header pointing to the data in the span
  return cv::Mat(1, span.size(), cv::traits::Type<T>::value,
                 const_cast<T *>(span.data())); // NOLINT
}
template <typename T> auto spanToMat1D(const std::span<T> &span) {
  // Create a Mat header pointing to the data in the span
  return cv::Mat(1, span.size(), cv::traits::Type<T>::value, span.data());
}

void conv1d_OpenCV(const cv::Mat &input, const cv::Mat &kernel,
                   cv::Mat &output) {
  // Ensure the kernel is either a single row or a single column
  CV_Assert(kernel.rows == 1 || kernel.cols == 1);

  // Use filter2D for convolution
  int ddepth = -1; // Keep the same depth as input
  cv::filter2D(input, output, ddepth, kernel, cv::Point(-1, -1), 0,
               cv::BORDER_CONSTANT);
}

template <typename T>
void conv1d_OpenCV(const std::span<const T> input,
                   const std::span<const T> kernel, std::span<T> output) {
  auto input_ = spanToMat1D(input);
  auto kernel_ = spanToMat1D(kernel);
  auto output_ = spanToMat1D(output);

  // Use filter2D for convolution
  int ddepth = -1; // Keep the same depth as input
  cv::filter2D(input_, output_, ddepth, kernel_, cv::Point(-1, -1), 0,
               cv::BORDER_CONSTANT);
}

/*
OpenCV Universal Intrinsics

https://docs.opencv.org/4.10.0/d6/dd1/tutorial_univ_intrin.html
https://docs.opencv.org/4.x/df/d91/group__core__hal__intrin.html
*/

template <typename T>
using cv_vector_type = typename std::conditional<
    std::is_same_v<T, float>, cv::v_float32,
    std::conditional_t<std::is_same_v<T, double>, cv::v_float64, void>>::type;

template <typename T>
void conv1d_OpenCV_intrin(const std::span<const T> input,
                          const std::span<const T> kernel,
                          std::span<T> output) {
  // OpenCV universal intrinsics support 128-bit SIMD
  // For 256-bit only AVX2 is supported
  using WideFloat = cv_vector_type<T>;
  int step = cv::VTraits<WideFloat>::vlanes();

  auto *sptr = input.data();
  auto *dptr = output.data();
  for (int k = 0; k < kernel.size(); ++k) {
    WideFloat kernel_wide = cv::vx_setall(kernel[k]);

    int i = 0;
    for (; i + step < input.size(); i += step) {
      WideFloat window = cv::vx_load(sptr + i + k);
      WideFloat sum =
          cv::v_add(cv::vx_load(dptr + i), cv::v_mul(kernel_wide, window));
      cv::v_store(dptr + i, sum);
    }

    for (; i < output.size(); ++i) {
      output[i] += input[i + k] * kernel[k];
    }
  }
}
