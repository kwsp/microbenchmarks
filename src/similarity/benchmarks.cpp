#include "similiarity.hpp"
#include <Eigen/Dense>
#include <armadillo>
#include <benchmark/benchmark.h>

void BM_similarity_naive(benchmark::State &state) {
  arma::Col<float> a(state.range(0), arma::fill::randn);
  arma::Col<float> b(state.range(0), arma::fill::randn);

  volatile auto result =
      cosine_similarity_naive(a.memptr(), b.memptr(), a.size());
  for (auto _ : state) {
    result = cosine_similarity_naive(a.memptr(), b.memptr(), a.size());
  }
}

BENCHMARK(BM_similarity_naive)->Range(4096, 16382);

#if defined(__AVX2__)

void BM_similarity_avx2(benchmark::State &state) {
  arma::Col<float> a(state.range(0), arma::fill::randn);
  arma::Col<float> b(state.range(0), arma::fill::randn);

  volatile auto result =
      cosine_similarity_avx2(a.memptr(), b.memptr(), a.size());
  for (auto _ : state) {
    result = cosine_similarity_avx2(a.memptr(), b.memptr(), a.size());
  }
}

BENCHMARK(BM_similarity_avx2)->Range(4096, 16382);

#endif

#if defined(__ARM_NEON__)

void BM_similarity_neon(benchmark::State &state) {
  arma::Col<float> a(state.range(0), arma::fill::randn);
  arma::Col<float> b(state.range(0), arma::fill::randn);

  volatile auto result =
      cosine_similarity_neon(a.memptr(), b.memptr(), a.size());
  for (auto _ : state) {
    result = cosine_similarity_neon(a.memptr(), b.memptr(), a.size());
  }
}

BENCHMARK(BM_similarity_neon)->Range(4096, 16382);

#endif

void BM_similarity_Eigen3(benchmark::State &state) {
  Eigen::VectorX<float> a(state.range(0));
  Eigen::VectorX<float> b(state.range(0));
  a.setRandom();
  b.setRandom();

  volatile auto result = cosine_similarity_Eigen(a, b);
  for (auto _ : state) {
    result = cosine_similarity_Eigen(a, b);
  }
}

BENCHMARK(BM_similarity_Eigen3)->Range(4096, 16382);

void BM_similarity_Arma(benchmark::State &state) {
  arma::Col<float> a(state.range(0), arma::fill::randn);
  arma::Col<float> b(state.range(0), arma::fill::randn);

  volatile auto result = cosine_similarity_Arma(a, b);
  for (auto _ : state) {
    result = cosine_similarity_Arma(a, b);
  }
}

BENCHMARK(BM_similarity_Arma)->Range(4096, 16382);