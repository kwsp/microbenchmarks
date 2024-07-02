#include <algorithm>
#include <armadillo>
#include <benchmark/benchmark.h>

// Helper function to swap blocks of columns
template <typename T>
void swap_blocks_cols(arma::Mat<T> &m, int start1, int start2, int block_size) {
  for (int i = 0; i < block_size; ++i) {
    m.swap_cols(start1 + i, start2 + i);
  }
}

// Helper function to swap blocks of rows
template <typename T>
void swap_blocks_rows(arma::Mat<T> &m, int start1, int start2, int block_size) {
  for (int i = 0; i < block_size; ++i) {
    m.swap_rows(start1 + i, start2 + i);
  }
}

// In-place shift for arma::mat using block swapping (shifts along specified
// dimension)
template <typename T> void shift_inplace(arma::Mat<T> &m, int shift, int dim) {
  if (dim == 0) // Shift rows
  {
    int n = m.n_rows;
    shift = shift % n; // To handle shifts greater than the number of rows
    if (shift < 0) {
      shift += n; // To handle negative shifts
    }

    if (shift == 0) {
      return;
    }

    // Block swapping for rows
    const int gcd = std::gcd(shift, n);
    for (int i = 0; i < gcd; ++i) {
      const int temp = m(i, 0);
      int j = i;

      while (true) {
        int k = j + shift;
        if (k >= n) {
          k -= n;
        }
        if (k == i) {
          break;
        }

        m.swap_rows(j, k);
        j = k;
      }
      m(j, 0) = temp;
    }
  } else if (dim == 1) // Shift columns
  {
    const int n = m.n_cols;
    shift = shift % n; // To handle shifts greater than the number of columns
    if (shift < 0) {
      shift += n; // To handle negative shifts
    }

    if (shift == 0) {
      return;
    }

    // Block swapping for columns
    const int gcd = std::gcd(shift, n);
    for (int i = 0; i < gcd; ++i) {
      int temp = m(0, i);
      int j = i;

      while (true) {
        int k = j + shift;
        if (k >= n) {
          k -= n;
        }
        if (k == i) {
          break;
        }

        m.swap_cols(j, k);
        j = k;
      }
      m(0, j) = temp;
    }
  } else {
    throw std::invalid_argument("Dimension must be 0 (rows) or 1 (columns).");
  }
}

template <typename T> auto shift_arma(arma::Mat<T> &m, int shift, int dim) {
  return arma::shift(m, shift, dim);
}

template <typename Func>
void BenchmarkFunc(benchmark::State &state, Func func) {
  arma::Mat<float> input(state.range(0), state.range(0), arma::fill::randu);
  for (auto _ : state) {
    arma::shift(input, 100, 1);
    // func(input, 100, 1);
  }
}

// Benchmark for Armadillo shift
static void BM_Shift(benchmark::State &state) {
  BenchmarkFunc(state, shift_arma<float>);
}
BENCHMARK(BM_Shift)->Range(256, 4096);

// Benchmark for inplace shift
static void BM_ShiftInplace(benchmark::State &state) {
  BenchmarkFunc(state, shift_inplace<float>);
}
BENCHMARK(BM_ShiftInplace)->Range(256, 4096);

BENCHMARK_MAIN();