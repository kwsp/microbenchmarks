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

// Function to convert matrix using a hand-rolled for loop
void OpenMPConversion(const arma::Mat<uint16_t> &input,
                      arma::Mat<double> &output) {
  output.set_size(input.n_rows, input.n_cols);
#pragma omp parallel for
  for (int i = 0; i < input.n_elem; ++i) {
    output.memptr()[i] = static_cast<double>(input.mem[i]);
  }
}

// Function to convert matrix using OpenCV's cv::parallel_for_
void OpenCVParallelConversion(const arma::Mat<uint16_t> &input,
                              arma::Mat<double> &output) {
  output.set_size(input.n_rows, input.n_cols);
  cv::parallel_for_(cv::Range(0, input.n_elem), [&](const cv::Range &range) {
    for (int i = range.start; i < range.end; ++i) {
      output.memptr()[i] = static_cast<double>(input.mem[i]);
    }
  });
}

const auto conversion_bench = [](const auto &callable) {
  return [&](benchmark::State &state) {
    arma::Mat<uint16_t> input(state.range(0), state.range(0),
                              arma::fill::randu);
    arma::Mat<double> output;
    for (auto _ : state) {
      callable(input, output);
    }
  };
};

// Benchmark for Armadillo conversion
static void BM_ArmadilloConversion(benchmark::State &state) {
  arma::Mat<uint16_t> input(state.range(0), state.range(0), arma::fill::randu);
  arma::Mat<double> output;
  for (auto _ : state) {
    ArmadilloConversion(input, output);
  }
}
BENCHMARK(BM_ArmadilloConversion)->Range(256, 4096);

// Benchmark for hand-rolled conversion
static void BM_HandRolledConversion(benchmark::State &state) {
  arma::Mat<uint16_t> input(state.range(0), state.range(0), arma::fill::randu);
  arma::Mat<double> output;
  for (auto _ : state) {
    HandRolledConversion(input, output);
  }
}
BENCHMARK(BM_HandRolledConversion)->Range(256, 4096);

// Benchmark for OpenMP conversion
static void BM_OpenMPConversion(benchmark::State &state) {
  arma::Mat<uint16_t> input(state.range(0), state.range(0), arma::fill::randu);
  arma::Mat<double> output;
  for (auto _ : state) {
    OpenMPConversion(input, output);
  }
}
BENCHMARK(BM_OpenMPConversion)->Range(256, 4096);

// Benchmark for OpenCV parallel_for_ conversion
static void BM_OpenCVParallelConversion(benchmark::State &state) {
  arma::Mat<uint16_t> input(state.range(0), state.range(0), arma::fill::randu);
  arma::Mat<double> output;
  for (auto _ : state) {
    OpenCVParallelConversion(input, output);
  }
}
BENCHMARK(BM_OpenCVParallelConversion)->Range(256, 4096);

BENCHMARK_MAIN();
