#include <armadillo>
#include <benchmark/benchmark.h>

template <typename T>
[[nodiscard]] arma::Mat<T> fast_shift_columns(const arma::Mat<T> &matrix,
                                              int shift) {
  arma::Mat<T> result(matrix.n_rows, matrix.n_cols, arma::fill::none);

  shift = shift % matrix.n_rows;
  if (shift < 0) {
    shift += matrix.n_rows; // Handle negative shifts
  }

  // Split and reorder rows
  result.rows(0, shift - 1) =
      matrix.rows(matrix.n_rows - shift, matrix.n_rows - 1);
  result.rows(shift, matrix.n_rows - 1) =
      matrix.rows(0, matrix.n_rows - shift - 1);

  return result;
}

template <typename T> auto shift_arma(arma::Mat<T> &m, int shift, int dim) {
  return arma::shift(m, shift, dim);
}

template <typename Func>
void BenchmarkFuncWithRet(benchmark::State &state, Func func) {
  arma::Mat<float> input(state.range(0), state.range(0), arma::fill::randu);
  for (auto _ : state) {
    volatile auto ret = func(input, 100, 1);
  }
}

template <typename Func>
void BenchmarkFuncNoRet(benchmark::State &state, Func func) {
  arma::Mat<float> input(state.range(0), state.range(0), arma::fill::randu);
  for (auto _ : state) {
    func(input, 100, 1);
  }
}

// Benchmark for Armadillo shift
static void BM_Shift(benchmark::State &state) {
  BenchmarkFuncWithRet(state, shift_arma<float>);
}
BENCHMARK(BM_Shift)->Range(256, 4096);

// Benchmark for inplace shift
static void BM_FastShiftColumns(benchmark::State &state) {
  arma::Mat<float> input(state.range(0), state.range(0), arma::fill::randu);
  for (auto _ : state) {
    volatile auto ret = fast_shift_columns(input, 100);
  }
}
BENCHMARK(BM_FastShiftColumns)->Range(256, 4096);

BENCHMARK_MAIN();