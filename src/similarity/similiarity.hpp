#pragma once

#include <cmath>
#include <cstddef>

// NOLINTBEGIN(*-isolate-declaration, *-pointer-arithmetic)
inline float cosine_similarity_naive(float const *a, float const *b, size_t n) {
  float dot{}, norm_a{}, norm_b{};
  for (int i = 0; i < n; ++i) {
    float ai = a[i], bi = b[i];
    dot += ai * bi;
    norm_a += ai * ai;
    norm_b += bi * bi;
  }
  if (dot == 0) return 0;
  if (norm_a == 0 || norm_b == 0) return 1;
  return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

#if defined(__AVX2__)

#include <immintrin.h>

// Accumulate the 8 f32 stored in a __m256 register in double precision
inline double reduce_f32x8_avx2(__m256 vec) {
  __m128 low_f32 = _mm256_castps256_ps128(vec);
  __m128 high_f32 = _mm256_extractf128_ps(vec, 1);
  __m256d low_f64 = _mm256_cvtps_pd(low_f32);
  __m256d high_f64 = _mm256_cvtps_pd(high_f32);
  __m256d sum = _mm256_add_pd(low_f64, high_f64);
  __m128d sum_low = _mm256_castpd256_pd128(sum);
  __m128d sum_high = _mm256_extractf128_pd(sum, 1);
  __m128d sum128 = _mm_add_pd(sum_low, sum_high);
  sum128 = _mm_hadd_pd(sum128, sum128);
  return _mm_cvtsd_f64(sum128);
}

inline __m256 partial_load_f32_avx2(float const *a, size_t n) {
  union {
    __m256 vec;
    float scalars[8];
  } result;
  for (size_t i = 0; i < n; ++i)
    result.scalars[i] = a[i];
  return result.vec;
}

// Approximate reciprocal square root
// Cosine normalize
// result = ab / (sqrt(a2) * sqrt(b2))
inline double cos_normalize_f64_avx2(double ab, double a2, double b2) {
  if (ab == 0) return 0;
  if (a2 == 0 || b2 == 0) return 1;

  __m128d squares = _mm_set_pd(a2, b2);

  // _mm_rsqrt_ps can introduce errors as high as 1.5*2^-12 ~= 3.66e-4
  __m128d rsqrts = _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(squares)));

  // https://web.archive.org/web/20210208132927/http://assemblyrequired.crashworks.org/timing-square-root/
  rsqrts = _mm_add_pd(
      _mm_mul_pd(_mm_set1_pd(1.5), rsqrts),
      _mm_mul_pd(_mm_mul_pd(_mm_mul_pd(squares, _mm_set1_pd(-0.5)), rsqrts),
                 _mm_mul_pd(rsqrts, rsqrts)));

  double a2_reciprocal = _mm_cvtsd_f64(_mm_unpackhi_pd(rsqrts, rsqrts));
  double b2_reciprocal = _mm_cvtsd_f64(rsqrts);
  return ab * a2_reciprocal * b2_reciprocal;
}

inline double cosine_similarity_avx2(float const *a, float const *b, size_t n) {
  __m256 a_vec, b_vec;
  __m256 ab_vec = _mm256_setzero_ps();
  __m256 a2_vec = _mm256_setzero_ps(), b2_vec = _mm256_setzero_ps();

  // 32 bytes in 256-bits
  constexpr size_t n_step = 32 / sizeof(float);

cosine_similarity_avx2_cycle:
  if (n < 8) {
    // Handle input length that aren't a multiple of 8
    a_vec = partial_load_f32_avx2(a, n);
    b_vec = partial_load_f32_avx2(b, n);
    n = 0;
  } else {
    // Load 8
    a_vec = _mm256_load_ps(a);
    b_vec = _mm256_load_ps(b);
    n -= n_step, a += n_step, b += n_step;
  }

  // Multiply and add to the acc
  ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);
  a2_vec = _mm256_fmadd_ps(a_vec, a_vec, a2_vec);
  b2_vec = _mm256_fmadd_ps(b_vec, b_vec, b2_vec);

  if (n) goto cosine_similarity_avx2_cycle;

  auto ab = reduce_f32x8_avx2(ab_vec);
  auto a2 = reduce_f32x8_avx2(a2_vec);
  auto b2 = reduce_f32x8_avx2(b2_vec);

  return cos_normalize_f64_avx2(ab, a2, b2);
}

#endif

#if defined(__ARM_NEON__)

#include <arm_neon.h>

double cos_normalize_f64_neon(double ab, double a2, double b2) {
  if (ab == 0) return 0;
  if (a2 == 0 || b2 == 0) return 1;

  double squares_arr[2] = {a2, b2};
  float64x2_t squares = vld1q_f64(squares_arr);

  // Arm NEON manuals don't explicitly mention the accuracy of `rsqrt`
  // Third party results suggests its less accurate than SSE errors (1.5*2^-12).
  // Use Newton-Raphson refinement
  float64x2_t rsqrts = vrsqrteq_f64(squares);
  // 2 rounds of Newton-Raphson refinement
  rsqrts = vmulq_f64(rsqrts, vrsqrtsq_f64(vmulq_f64(squares, rsqrts), rsqrts));
  rsqrts = vmulq_f64(rsqrts, vrsqrtsq_f64(vmulq_f64(squares, rsqrts), rsqrts));
  vst1q_f64(squares_arr, rsqrts);
  double result = 1 - ab * squares_arr[0] * squares_arr[1];
  return result > 0 ? result : 0;
}

inline double cosine_similarity_neon(float const *a, float const *b, size_t n) {
  float32x4_t ab_vec = vdupq_n_f32(0);
  float32x4_t a2_vec = vdupq_n_f32(0), b2_vec = vdupq_n_f32(0);
  size_t i = 0;
  constexpr size_t n_step = 128 / 32;
  for (; i + n_step <= n; i += n_step) {
    float32x4_t a_vec = vld1q_f32(a + i);
    float32x4_t b_vec = vld1q_f32(b + i);
    ab_vec = vfmaq_f32(a_vec, b_vec, ab_vec);
    a2_vec = vfmaq_f32(a_vec, a_vec, a2_vec);
    b2_vec = vfmaq_f32(b_vec, b_vec, b2_vec);
  }
  float ab = vaddvq_f32(ab_vec);
  float a2 = vaddvq_f32(a2_vec), b2 = vaddvq_f32(b2_vec);
  for (; i < n; ++i) {
    float ai = a[i], bi = b[i];
    ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
  }

  return cos_normalize_f64_neon(ab, a2, b2);
}

#endif

// NOLINTEND(*-isolate-declaration, *-pointer-arithmetic)

#include <Eigen/Dense>

double cosine_similarity_Eigen(const Eigen::VectorX<float> &a,
                               const Eigen::VectorX<float> &b) {
  double dot = a.dot(b);
  double norm_a = a.norm();
  double norm_b = b.norm();
  if (norm_a == 0 || norm_b == 0) return 0;
  if (dot == 0) return 1;
  return dot / (norm_a * norm_b);
}

#include <Armadillo>

double cosine_similarity_Arma(const arma::Col<float> &a,
                              const arma::Col<float> &b) {
  auto dot = arma::dot(a, b);
  auto norm_a = arma::norm(a);
  auto norm_b = arma::norm(b);
  if (norm_a == 0 || norm_b == 0) return 0;
  if (dot == 0) return 1;
  return dot / (norm_a * norm_b);
}