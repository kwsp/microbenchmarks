#include <benchmark/benchmark.h>
#include <cmath>
#include <fftconv/aligned_vector.hpp>
#include <fftconv/hilbert.hpp>
#include <numbers>

// NOLINTBEGIN(*-magic-numbers)

using fftconv::AlignedVector;

template <typename T, typename Func>
void hilbert_bench(benchmark::State &state, Func hilbert_func) {

  const auto N = state.range(0);
  AlignedVector<T> in(N);
  for (int i = 0; i < N; ++i) {
    in[i] = std::cos(std::numbers::pi_v<T> * 4 * i / (N - 1));
  }
  AlignedVector<T> out(N);

  hilbert_func(in, out);
  for (auto _ : state) {
    hilbert_func(in, out);
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(T));
}

template <typename T> void BM_hilbert_fftw(benchmark::State &state) {
  hilbert_bench<T>(state, fftconv::hilbert<T>);
}
BENCHMARK(BM_hilbert_fftw<float>)->DenseRange(2048, 6144, 1024);
BENCHMARK(BM_hilbert_fftw<double>)->DenseRange(2048, 6144, 1024);

// template <typename T> void BM_hilbert_fftw_split(benchmark::State &state) {
//   hilbert_bench<T>(state, hilbert_fftw_split<T>);
// }
// BENCHMARK(BM_hilbert_fftw_split<float>)->DenseRange(2048,6144,1024);
// BENCHMARK(BM_hilbert_fftw_split<double>)->DenseRange(2048,6144,1024);

#if defined(HAS_IPP)

template <typename T> void BM_hilbert_ipp(benchmark::State &state) {
  hilbert_bench<T>(state, hilbert_ipp<T>);
}
BENCHMARK(BM_hilbert_ipp<float>)->DenseRange(2048, 6144, 1024);
BENCHMARK(BM_hilbert_ipp<double>)->DenseRange(2048, 6144, 1024);

#endif

int main(int argc, char **argv) {
  fftw::WisdomSetup _fftwSetup(true);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}

// NOLINTEND(*-magic-numbers)