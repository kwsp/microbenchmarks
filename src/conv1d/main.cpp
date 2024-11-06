#include "conv1d.hpp"

// NOLINTBEGIN(*-magic-numbers)

int main(int argc, char *argv[]) {
  using T = double;
  std::vector<T> input = {1, 2, 3, 4, 5, 6};
  std::vector<T> kernel = {1, 0, 1};

  auto output_size_valid = input.size() - kernel.size() + 1;
  auto output_size_same = input.size();
  auto output_size_full = input.size() + kernel.size() - 1;

  {
    std::vector<T> output(output_size_valid, 0);
    std::vector<T> im2col_mat(output_size_valid * kernel.size(), 0);
    conv1d_BLAS_im2col<T>(input, kernel, im2col_mat, output);
    fmt::println("=== BLAS (im2col + gemm) ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }

#ifdef __APPLE__
  {
    std::vector<T> output(output_size_valid, 0);
    conv1d_vDSP<T>(input, kernel, output);
    fmt::println("=== Accelerate vDSP ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }
#endif

  {
    std::vector<T> output(output_size_valid, 0);
    conv1d_eigen<T>(input, kernel, output);
    fmt::println("=== Eigen ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }

  {
    std::vector<T> output(output_size_same, 0);
    conv1d_KFR_fir<T>(input, kernel, output);
    fmt::println("=== KFR (FIR convolve) ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }

#ifndef __APPLE__
  {
    std::vector<T> output(output_size_same, 0);
    conv1d_kfr_oa<T>(input, kernel, output);
    fmt::println("=== KFR (oa) ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }
#endif

  {
    std::vector<T> output(output_size_same, 0);
    conv1d_OpenCV<T>(input, kernel, output);
    fmt::println("=== OpenCV ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }

  return 0;
}

// NOLINTEND(*-magic-numbers)