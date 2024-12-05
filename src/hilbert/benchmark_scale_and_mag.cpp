#include <benchmark/benchmark.h>

#include "aligned_vector.hpp"
#include "fftw.hpp"

template <typename T, typename Func>
void ScaleAndMag(benchmark::State &state, Func func) {
  const auto N = state.range(0);

  auto *in = fftw::alloc_complex<T>(N);
  AlignedVector<T> out(N);
  const T fct = 0.5;
  func(in, out.data(), N, fct);
  for (auto _ : state) {
    func(in, out.data(), N, fct);
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(T));

  fftw::free<T>(in);
}

template <typename T> void BM_ScaleAndMag_serial(benchmark::State &state) {
  ScaleAndMag<T>(state, fftw::scale_and_magnitude_serial<T>);
}
BENCHMARK(BM_ScaleAndMag_serial<float>)->Range(2048, 8192);
BENCHMARK(BM_ScaleAndMag_serial<double>)->Range(2048, 8192);

#if defined(__AVX2__)

template <typename T> void BM_ScaleAndMag_avx2(benchmark::State &state) {
  ScaleAndMag<T>(state, fftw::scale_and_magnitude_avx2<T>);
}
BENCHMARK(BM_ScaleAndMag_avx2<float>)->Range(2048, 8192);
BENCHMARK(BM_ScaleAndMag_avx2<double>)->Range(2048, 8192);

#endif

BENCHMARK_MAIN();