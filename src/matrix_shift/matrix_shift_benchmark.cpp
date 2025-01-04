#include <armadillo>
#include <benchmark/benchmark.h>

template <typename T>
[[nodiscard]] arma::Mat<T> fast_shift_columns(const arma::Mat<T> &mat,
                                              int shift) {
  arma::Mat<T> result(mat.n_rows, mat.n_cols, arma::fill::none);

  shift = shift % mat.n_rows;
  if (shift < 0) {
    shift += mat.n_rows; // Handle negative shifts
  }

  // Split and reorder rows
  result.rows(0, shift - 1) = mat.rows(mat.n_rows - shift, mat.n_rows - 1);
  result.rows(shift, mat.n_rows - 1) = mat.rows(0, mat.n_rows - shift - 1);

  return result;
}

/**
 * Assume the input is continuous!
 */
template <typename T> void fast_shift_columns2(arma::Mat<T> &mat, int shift) {

  shift = shift % mat.n_rows;
  if (shift < 0) {
    shift += mat.n_rows; // Handle negative shifts
  }

  size_t shift_n = shift * mat.n_cols;
  std::rotate(mat.begin(), mat.begin() + shift_n, mat.end());
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

static void BM_FastShiftColumns(benchmark::State &state) {
  arma::Mat<float> input(state.range(0), state.range(0), arma::fill::randu);
  for (auto _ : state) {
    volatile auto ret = fast_shift_columns(input, 100);
  }
}
BENCHMARK(BM_FastShiftColumns)->Range(256, 4096);

static void BM_FastShiftColumns2(benchmark::State &state) {
  arma::Mat<float> input(state.range(0), state.range(0), arma::fill::randu);
  for (auto _ : state) {
    fast_shift_columns2(input, 100);
  }
}
BENCHMARK(BM_FastShiftColumns2)->Range(256, 4096);

BENCHMARK_MAIN();