#include <armadillo>
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>
#include <vector>

// Function to convert matrix using Armadillo's conv_to
void ArmadilloConversion(const arma::Mat<uint16_t> &input,
                         arma::Mat<double> &output) {
  output = arma::conv_to<arma::Mat<double>>::from(input);
}

// Function to convert matrix using a hand-rolled for loop
void HandRolledConversion(const arma::Mat<uint16_t> &input,
                          arma::Mat<double> &output) {
  output.set_size(input.n_rows, input.n_cols);
  for (size_t i = 0; i < input.n_elem; ++i) {
    output.memptr()[i] = static_cast<double>(input.mem[i]);
  }
}

#ifdef HAS_OMP
// Function to convert matrix using a hand-rolled for loop
void OpenMPConversion(const arma::Mat<uint16_t> &input,
                      arma::Mat<double> &output) {
  output.set_size(input.n_rows, input.n_cols);
#pragma omp parallel for
  for (int i = 0; i < input.n_elem; ++i) {
    output.memptr()[i] = static_cast<double>(input.mem[i]);
  }
}
#endif

// Function to convert matrix using OpenCV's cv::parallel_for_
void OpenCVParallelConversion1(const arma::Mat<uint16_t> &input,
                               arma::Mat<double> &output) {
  output.set_size(input.n_rows, input.n_cols);
  cv::parallel_for_(cv::Range(0, input.n_elem), [&](const cv::Range &range) {
    for (int i = range.start; i < range.end; ++i) {
      output.memptr()[i] = static_cast<double>(input.mem[i]);
    }
  });
}

// Function to convert matrix using OpenCV's cv::parallel_for_
void OpenCVParallelConversion2(const arma::Mat<uint16_t> &input,
                               arma::Mat<double> &output) {
  output.set_size(input.n_rows, input.n_cols);
  cv::parallel_for_(cv::Range(0, input.n_cols), [&](const cv::Range &range) {
    for (int col = range.start; col < range.end; ++col) {
      const auto *inptr = input.colptr(col);
      auto *outptr = output.colptr(col);
      for (int i = 0; i < input.n_rows; ++i) {
        outptr[i] = static_cast<double>(inptr[i]);
      }
    }
  });
}

void OpenCVMatConversion(const cv::Mat &input, cv::Mat &output) {
  input.convertTo(output, CV_64F);
}

void OpenCVMatParallelConversion(const cv::Mat &input, cv::Mat &output) {
  if (output.rows != input.rows || output.cols != input.cols ||
      output.type() != CV_64F) {
    input.convertTo(output, CV_64F);
    return;
  }

  // Use OpenCV's parallel_for_ to parallelize the conversion
  cv::parallel_for_(cv::Range(0, input.rows), [&](const cv::Range &range) {
    for (int i = range.start; i < range.end; ++i) {
      const uint16_t *inputRowPtr = input.ptr<uint16_t>(i);
      double *outputRowPtr = output.ptr<double>(i);
      for (int j = 0; j < input.cols; ++j) {
        outputRowPtr[j] = static_cast<double>(inputRowPtr[j]);
      }
    }
  });
}

template <typename Func>
void BenchmarkArmaFunc(benchmark::State &state, Func func) {
  arma::Mat<uint16_t> input(state.range(0), state.range(0), arma::fill::randu);
  arma::Mat<double> output;
  for (auto _ : state) {
    func(input, output);
  }
}

// Benchmark for Armadillo conversion
static void BM_ArmadilloConversion(benchmark::State &state) {
  BenchmarkArmaFunc(state, ArmadilloConversion);
}
BENCHMARK(BM_ArmadilloConversion)->Range(256, 4096);

// Benchmark for hand-rolled conversion
static void BM_HandRolledConversion(benchmark::State &state) {
  BenchmarkArmaFunc(state, HandRolledConversion);
}
BENCHMARK(BM_HandRolledConversion)->Range(256, 4096);

#ifdef HAS_OMP
// Benchmark for OpenMP conversion
static void BM_OpenMPConversion(benchmark::State &state) {
  BenchmarkArmaFunc(state, OpenMPConversion);
}
BENCHMARK(BM_OpenMPConversion)->Range(256, 4096);
#endif

// Benchmark for OpenCV parallel_for_ conversion
static void BM_OpenCVParallelConversion1(benchmark::State &state) {
  BenchmarkArmaFunc(state, OpenCVParallelConversion1);
}
BENCHMARK(BM_OpenCVParallelConversion1)->Range(256, 4096);

static void BM_OpenCVParallelConversion2(benchmark::State &state) {
  BenchmarkArmaFunc(state, OpenCVParallelConversion2);
}
BENCHMARK(BM_OpenCVParallelConversion2)->Range(256, 4096);

template <typename Func>
void BenchmarkCvFunc(benchmark::State &state, Func func) {
  cv::Mat input(state.range(0), state.range(0), CV_16UC1);
  cv::randu(input, 0, 1);
  cv::Mat output(state.range(0), state.range(0), CV_64F);
  for (auto _ : state) {
    func(input, output);
  }
}

// Benchmark for OpenCV parallel_for_ conversion
static void BM_OpenCVMatConversion(benchmark::State &state) {
  BenchmarkCvFunc(state, OpenCVMatConversion);
}
BENCHMARK(BM_OpenCVMatConversion)->Range(256, 4096);

// Benchmark for OpenCV parallel_for_ conversion
static void BM_OpenCVMatParallelConversion(benchmark::State &state) {
  BenchmarkCvFunc(state, OpenCVMatParallelConversion);
}
BENCHMARK(BM_OpenCVMatParallelConversion)->Range(256, 4096);

BENCHMARK_MAIN();
