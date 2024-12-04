#include "hilbert.hpp"
#include <Eigen/Dense>
#include <armadillo>
#include <benchmark/benchmark.h>

// NOLINTBEGIN(*-magic-numbers)

template <typename T, typename Func>
void hilbert_bench(benchmark::State &state, Func hilbert_func) {
  Eigen::VectorX<T> inp(state.range(0));
  inp.setRandom();
  Eigen::VectorX<T> out(state.range(0));

  std::span<const T> inp_(inp.data(), static_cast<size_t>(inp.size()));
  std::span<T> out_(out.data(), static_cast<size_t>(out.size()));

  hilbert_func(inp_, out_);
  for (auto _ : state) {
    hilbert_func(inp_, out_);
  }
}

template <typename T> void BM_hilbert_fftw(benchmark::State &state) {
  hilbert_bench<T>(state, hilbert_abs<T>);
}
BENCHMARK(BM_hilbert_fftw<float>)->Range(2048, 8192);
BENCHMARK(BM_hilbert_fftw<double>)->Range(2048, 8192);

template <typename T> void BM_hilbert_fftw_split(benchmark::State &state) {
  hilbert_bench<T>(state, hilbert_abs_2<T>);
}
BENCHMARK(BM_hilbert_fftw_split<float>)->Range(2048, 8192);
BENCHMARK(BM_hilbert_fftw_split<double>)->Range(2048, 8192);

int main(int argc, char **argv) {
  fftw::WisdomSetup _fftwSetup(true);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}

// NOLINTEND(*-magic-numbers)