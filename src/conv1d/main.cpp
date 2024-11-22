#include "conv1d.hpp"
#include "fftconv.hpp"
#include <fftw3.h>

#ifdef CONV1D_HAS_IPP
#include <ipp.h>
#endif

// NOLINTBEGIN(*-magic-numbers)

int main(int argc, char *argv[]) {

#ifdef CONV1D_HAS_IPP
  ippInit();
  get_ipp_version();
#endif

  if (fftw_init_threads() != 0) {
    fmt::println("fftw_init_threads failed.");
  }
  if (fftwf_init_threads() != 0) {
    fmt::println("fftwf_init_threads failed.");
  }
  fftw_plan_with_nthreads(8);
  fftwf_plan_with_nthreads(8);

  using T = double;

  const std::vector<T> input = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<T> kernel = {1, 0, 2, 1};

  auto output_size_valid = input.size() - kernel.size() + 1;
  auto output_size_same = input.size();
  auto output_size_full = input.size() + kernel.size() - 1;

  {
    std::vector<T> output(output_size_full, 0);
    conv1d_naive<T, ConvMode::Full>(input, kernel, output);
    fmt::println("=== Naive (full) ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }

  {
    std::vector<T> output(output_size_same, 0);
    conv1d_naive<T, ConvMode::Same>(input, kernel, output);
    fmt::println("=== Naive (same) ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }

  {
    std::vector<T> output(output_size_valid, 0);
    conv1d_naive<T, ConvMode::Valid>(input, kernel, output);
    fmt::println("=== Naive (valid) ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }

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

  // #ifndef __APPLE__
  //   {
  //     std::vector<T> output(output_size_same, 0);
  //     conv1d_kfr_oa<T>(input, kernel, output);
  //     fmt::println("=== KFR (oa) ===");
  //     fmt::println("Output: {}", fmt::join(output, ", "));
  //   }
  // #endif

  {
    std::vector<T> output(output_size_same, 0);
    conv1d_OpenCV<T>(input, kernel, output);
    fmt::println("=== OpenCV ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }

  {
    std::vector<T> output(output_size_valid, 0);
    conv1d_OpenCV_intrin<T>(input, kernel, output);
    fmt::println("=== OpenCV (Universal Intrinsics) ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }

  {
    // TODO run ASAN
    std::vector<T> output(output_size_same, 0);
    fftconv::oaconvolve_fftw_same<T>(input, kernel, output);
    fmt::println("=== fftconv (oa, same) ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }

#ifdef CONV1D_HAS_IPP

  {
    std::vector<T> output(output_size_full, 0);
    conv1d_IPP<T, IppAlgType::ippAlgDirect>(input, kernel, output);
    fmt::println("=== Intel IPP (direct) ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }
  {
    std::vector<T> output(output_size_full, 0);
    conv1d_IPP<T, IppAlgType::ippAlgFFT>(input, kernel, output);
    fmt::println("=== Intel IPP (FFT) ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }

#endif

  return 0;
}

// NOLINTEND(*-magic-numbers)