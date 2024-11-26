#include "conv1d.hpp"
#include <armadillo>
#include <benchmark/benchmark.h>
#include <fftconv.hpp>
#include <fftw.hpp>
#include <utility>
#include <vector>

#define RUN_ALL

// // NOLINTNEXTLIE(*-magic-numbers)
// const std::vector<std::pair<int64_t, int64_t>> RANGES{
//     {{2 << 7, 2 << 12}, {15, 95}},
// };

// const std::vector<std::vector<int64_t>> ARGS{
//     {
//         2048,
//         4096,
//         8192,
//     },
//     {
//         65,
//         95,
//         145,
//     },
// };

const std::vector<std::vector<int64_t>> ARGS{
    {
        2048,
        3072,
        4096,
        5120,
        6144,
        7168,
        8192,
        9216,
    },
    {
        65,
        // 85,
        // 105,
        // 125,
        // 145,
        165,
        // 185,
        // 205,
        // 225,
        245,
    },
};

// NOLINTNEXTLIE(*-magic-numbers)
// const std::vector<std::pair<int64_t, int64_t>> RANGES{
//     {{2 << 11, 2 << 11}, {65, 65}}};

/*
Helpers
*/
template <typename T, typename Func>
void conv_bench_full(benchmark::State &state, Func conv_func) {
  arma::Col<T> input(state.range(0), arma::fill::randn);
  arma::Col<T> kernel(state.range(1), arma::fill::randn);
  arma::Col<T> output(input.size() + kernel.size() - 1);

  conv_func(input, kernel, output);
  for (auto _ : state) {
    conv_func(input, kernel, output);
  }
}

template <typename T, typename Func>
void conv_bench_same(benchmark::State &state, Func conv_func) {
  arma::Col<T> input(state.range(0), arma::fill::randn);
  arma::Col<T> kernel(state.range(1), arma::fill::randn);
  arma::Col<T> output(input.size());

  conv_func(input, kernel, output);
  for (auto _ : state) {
    conv_func(input, kernel, output);
  }
}

template <typename T, typename Func>
void conv_bench_valid(benchmark::State &state, Func conv_func) {
  arma::Col<T> input(state.range(0), arma::fill::randn);
  arma::Col<T> kernel(state.range(1), arma::fill::randn);
  arma::Col<T> output(input.size() - kernel.size() + 1);

  conv_func(input, kernel, output);
  for (auto _ : state) {
    conv_func(input, kernel, output);
  }
}

#ifdef RUN_ALL

/*
Conv1d naive
*/
template <typename T> static void BM_conv1d_naive(benchmark::State &state) {
  conv_bench_valid<T>(state, conv1d_naive<T, ConvMode::Valid>);
}
BENCHMARK(BM_conv1d_naive<double>)->ArgsProduct(ARGS);

/*
Conv1d with BLAS
*/
template <typename T> static void BM_conv1d_BLAS(benchmark::State &state) {
  arma::Col<T> input(state.range(0), arma::fill::randn);
  arma::Col<T> kernel(state.range(1), arma::fill::randn);
  arma::Col<T> output(input.size() - kernel.size() + 1);

  arma::Col<T> im2col(output.size() * kernel.size());

  for (auto _ : state) {
    conv1d_BLAS_im2col<T>(input, kernel, im2col, output);
  }
}
BENCHMARK(BM_conv1d_BLAS<double>)->ArgsProduct(ARGS);

// Wrapper to prevent arma::conv from being optimized away
// Not storing results back in res.
template <fftconv::FloatOrDouble Real>
void arma_conv_full(const std::span<const Real> span1,
                    const std::span<const Real> span2,
                    std::span<Real> span_res) {
  // NOLINTBEGIN(*-const-cast)
  const arma::Col<Real> vec1(const_cast<Real *>(span1.data()), span1.size(),
                             false, true);
  const arma::Col<Real> vec2(const_cast<Real *>(span2.data()), span2.size(),
                             false, true);
  // NOLINTEND(*-const-cast)
  volatile arma::Col<Real> res = arma::conv(vec1, vec2);
}

template <fftconv::FloatOrDouble Real>
void BM_conv1d_Arma(benchmark::State &state) {
  conv_bench_full<Real>(state, arma_conv_full<Real>);
}
BENCHMARK(BM_conv1d_Arma<double>)->ArgsProduct(ARGS);

template <fftconv::FloatOrDouble Real>
void BM_conv1d_Eigen(benchmark::State &state) {
  conv_bench_full<Real>(state, conv1d_eigen<Real>);
}
BENCHMARK(BM_conv1d_Eigen<double>)->ArgsProduct(ARGS);

template <fftconv::FloatOrDouble Real>
void BM_conv1d_KFR(benchmark::State &state) {
  conv_bench_same<Real>(state, conv1d_KFR_fir<Real>);
}
BENCHMARK(BM_conv1d_KFR<double>)->ArgsProduct(ARGS);

template <fftconv::FloatOrDouble Real>
void BM_conv1d_OpenCV(benchmark::State &state) {
  conv_bench_same<Real>(state, conv1d_OpenCV<Real>);
}
BENCHMARK(BM_conv1d_OpenCV<double>)->ArgsProduct(ARGS);

// template <fftconv::FloatOrDouble Real>
// void BM_conv1d_OpenCV_intrin(benchmark::State &state) {
//   conv_bench_same<Real>(state, conv1d_OpenCV_intrin<Real>);
// }
// BENCHMARK(BM_conv1d_OpenCV_intrin<double>)->ArgsProduct(ARGS);

#endif // RUN_ALL

// template <fftconv::FloatOrDouble Real>
// void BM_conv1d_fftconv(benchmark::State &state) {
//   conv_bench_full<Real>(state, fftconv::convolve_fftw<Real>);
// }
// BENCHMARK(BM_conv1d_fftconv<double>)->ArgsProduct(ARGS);

template <fftconv::FloatOrDouble Real>
void BM_conv1d_fftconv_oa(benchmark::State &state) {
  conv_bench_full<Real>(state, fftconv::oaconvolve_fftw<Real>);
}
BENCHMARK(BM_conv1d_fftconv_oa<double>)->ArgsProduct(ARGS);

template <fftconv::FloatOrDouble Real>
void BM_conv1d_fftconv_oa_same(benchmark::State &state) {
  conv_bench_same<Real>(state, fftconv::oaconvolve_fftw_same<Real>);
}
// BENCHMARK(BM_conv1d_fftconv_oa_same<double>)->ArgsProduct(ARGS);

#ifdef CONV1D_HAS_IPP

template <fftconv::FloatOrDouble Real>
void BM_conv1d_ipp_full_direct(benchmark::State &state) {
  conv_bench_full<Real>(state, conv1d_IPP<Real, IppAlgType::ippAlgDirect>);
}
BENCHMARK(BM_conv1d_ipp_full_direct<double>)->ArgsProduct(ARGS);

template <fftconv::FloatOrDouble Real>
void BM_conv1d_ipp_full_fft(benchmark::State &state) {
  conv_bench_full<Real>(state, conv1d_IPP<Real, IppAlgType::ippAlgFFT>);
}
BENCHMARK(BM_conv1d_ipp_full_fft<double>)->ArgsProduct(ARGS);

#endif

#ifdef __APPLE__

/*
Conv1d with Accelerate
*/
template <typename T>
static void BM_conv1d_Accelerate_vDSP(benchmark::State &state) {
  conv_bench_valid<T>(state, conv1d_vDSP<T>);
}
BENCHMARK(BM_conv1d_Accelerate_vDSP<double>)->ArgsProduct(ARGS);

#endif

int main(int argc, char **argv) {
  fftw::FFTWGlobalSetup _fftwSetup;

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
