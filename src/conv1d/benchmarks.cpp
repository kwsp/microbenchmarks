#include "conv1d.hpp"
#include <armadillo>
#include <benchmark/benchmark.h>
#include <utility>
#include <vector>

// NOLINTNEXTLIE(*-magic-numbers)
const std::vector<std::pair<int64_t, int64_t>> RANGES{
    {{2 << 7, 2 << 12}, {15, 95}}};

/*
Conv1d with BLAS
*/
template <typename T> static void BM_conv1d_BLAS(benchmark::State &state) {
  arma::Col<T> input(state.range(0), arma::fill::randn);
  arma::Col<T> kernel(state.range(0), arma::fill::randn);

  int output_size = input.size() - kernel.size() + 1;
  arma::Col<T> output(output_size, arma::fill::zeros);

  std::vector<T> im2col_matrix(output_size * kernel.size());

  for (auto _ : state) {
    conv1d_openblas<T>(input, kernel, im2col_matrix, output);
  }
}
BENCHMARK(BM_conv1d_BLAS<double>)->Ranges(RANGES);

#ifdef __APPLE__
/*
Conv1d with Accelerate
*/
template <typename T> static void BM_conv1d_vdsp(benchmark::State &state) {
  arma::Col<T> input(state.range(0), arma::fill::randn);
  arma::Col<T> kernel(state.range(0), arma::fill::randn);

  int output_size = input.size() - kernel.size() + 1;
  arma::Col<T> output(output_size, arma::fill::zeros);

  for (auto _ : state) {
    conv1d_vdsp<T>(input, kernel, output);
  }
}
BENCHMARK(BM_conv1d_vdsp<double>)->Ranges(RANGES);

#endif

BENCHMARK_MAIN();