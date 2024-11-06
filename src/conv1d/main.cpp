#include "conv1d.hpp"
#include <algorithm>

int main(int argc, char *argv[]) {
  using T = double;
  std::vector<T> input = {1, 2, 3, 4, 5, 6};
  std::vector<T> kernel = {1, 0, 1};

  auto output_size = input.size() - kernel.size() + 1;
  std::vector<T> output(output_size, 0);
  std::vector<T> im2col_mat(output_size * kernel.size(), 0);

  {
    conv1d_openblas<T>(input, kernel, im2col_mat, output);
    // auto output = conv1d_openblas_same<T>(input, kernel);
    fmt::println("=== BLAS (im2col + gemm) ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }

#ifdef __APPLE__
  {
    // auto output = conv1d_openblas(input, kernel);
    std::fill(output.begin(), output.end(), 0);
    conv1d_vdsp<T>(input, kernel, output);
    fmt::println("=== Accelerate vDSP ===");
    fmt::println("Output: {}", fmt::join(output, ", "));
  }
#endif

  return 0;
}
