#pragma once

#include "fftw.hpp"
#include <cassert>
#include <span>

// NOLINTBEGIN(*-pointer-arithmetic, *-magic-numbers)

/**
@brief Compute the analytic signal, using the Hilbert transform.
*/
template <fftw::Floating T>
void hilbert_abs(const std::span<const T> x, const std::span<T> env) {
  const auto n = x.size();
  assert(n > 0);
  assert(x.size() == env.size());

  auto &engine = fftw::EngineDFT1D<T>::get(n);
  auto &buf = engine.buf;

  // Copy input to real buffer
  for (int i = 0; i < n; ++i) {
    buf.in[i][0] = x[i];
    buf.in[i][1] = 0.;
  }

  // Execute r2c fft
  engine.forward();

  // Zero negative frequencies (half-Hermitian to Hermitian conversion)
  // Double the magnitude of positive frequencies
  const auto n_half = n / 2;
  for (auto i = 1; i < n_half; ++i) {
    buf.out[i][0] *= 2.;
    buf.out[i][1] *= 2.;
  }

  if (n % 2 == 0) {
    buf.out[n_half][0] = 0.;
    buf.out[n_half][1] = 0.;
  } else {
    buf.out[n_half][0] *= 2.;
    buf.out[n_half][1] *= 2.;
  }

  for (auto i = n_half + 1; i < n; ++i) {
    buf.out[i][0] = 0.;
    buf.out[i][1] = 0.;
  }

  // Execute c2r fft on modified spectrum
  engine.backward();

  // Take the abs of the analytic signal
  const T fct = static_cast<T>(1. / n);
  for (auto i = 0; i < n; ++i) {
    const auto real = x[i];
    const auto imag = buf.in[i][1] * fct;
    const auto res = std::sqrt(real * real + imag * imag);
    env[i] = res;
  }
}

template <typename T>
void hilbert_abs_cx_scale_imag(T const *real, T const *imag, T fct, size_t n,
                               T *out);

/**
@brief Compute the analytic signal, using the Hilbert transform.
*/
template <fftw::Floating T>
void hilbert_abs_2(const std::span<const T> x, const std::span<T> env) {
  const size_t n = x.size();
  assert(n > 0);
  assert(x.size() == env.size());

  auto &engine = fftw::EngineDFTSplit1D<T>::get(n);
  auto &buf = engine.buf;

  // Copy input to real buffer
  std::copy(x.data(), x.data() + n, buf.ri);
  std::fill(buf.ii, buf.ii + n, 0);

  // Execute r2c fft
  engine.forward();

  // Zero negative frequencies (half-Hermitian to Hermitian conversion)
  // Double the magnitude of positive frequencies
  const size_t n_half = n / 2;

  for (auto i = 1; i < n_half; ++i)
    buf.ro[i] *= 2.0;

  for (auto i = 1; i < n_half; ++i) {
    buf.io[i] *= 2.0;
  }

  if (n % 2 == 0) {
    buf.ro[n_half] = 0.;
    buf.io[n_half] = 0.;
  } else {
    buf.ro[n_half] *= 2.;
    buf.io[n_half] *= 2.;
  }

  std::fill(buf.ro + n_half + 1, buf.ro + n, 0.);
  std::fill(buf.io + n_half + 1, buf.io + n, 0.);

  // Execute c2r fft on modified spectrum
  engine.backward();

  // Take the abs of the analytic signal
  const T fct = static_cast<T>(1. / n);
  hilbert_abs_cx_scale_imag(x.data(), buf.ii, fct, n, env.data());
}

#if defined(__ARM_NEON__)

#include <arm_neon.h>

template <typename T>
void hilbert_abs_cx_scale_imag_neon(T const *real, T const *imag, T fct,
                                    size_t n, T *out) {

  size_t i = 0;
  if constexpr (std::is_same_v<T, float>) {

    float32x4_t fct_vec = vdupq_n_f32(fct);
    constexpr size_t simd_width = 4;
    for (; i + simd_width <= n; i += simd_width) {
      auto real_vec1 = vld1q_f32(&real[i]);
      auto imag_vec1 = vld1q_f32(&imag[i]);

      // Proc first set
      imag_vec1 = vmulq_f32(imag_vec1, fct_vec);
      real_vec1 = vmulq_f32(real_vec1, real_vec1);
      imag_vec1 = vmulq_f32(imag_vec1, imag_vec1);
      auto sum_vec1 = vaddq_f32(real_vec1, imag_vec1);
      auto res1 = vsqrtq_f32(sum_vec1);
      vst1q_f32(&out[i], res1);
    }

  } else if constexpr (std::is_same_v<T, double>) {

    float64x2_t fct_vec = vdupq_n_f64(fct);
    constexpr size_t simd_width = 2;
    for (; i + simd_width <= n; i += simd_width) {
      auto real_vec = vld1q_f64(&real[i]);
      auto imag_vec = vld1q_f64(&imag[i]);

      // Scale imag
      imag_vec = vmulq_f64(imag_vec, fct_vec);
      // Square real and imag
      real_vec = vmulq_f64(real_vec, real_vec);
      imag_vec = vmulq_f64(imag_vec, imag_vec);

      // Sum the squares
      auto sum_vec = vaddq_f64(real_vec, imag_vec);
      // Sqrt
      auto res = vsqrtq_f64(sum_vec);
      vst1q_f64(&out[i], res);
    }

  } else {
    static_assert(false, "Not implemented.");
  }

  // Remaining
  for (; i < n; ++i) {
    const auto ri = real[i];
    const auto ii = imag[i] * fct;
    const auto res = std::sqrt(ri * ri + ii * ii);
    out[i] = res;
  }
}

#endif

template <typename T>
void hilbert_abs_cx_scale_imag(T const *real, T const *imag, T fct, size_t n,
                               T *out) {

#if defined(__ARM_NEON__)

  hilbert_abs_cx_scale_imag_neon<T>(real, imag, fct, n, out);

#else

  for (auto i = 0; i < n; ++i) {
    const auto ri = real[i];
    const auto ii = imag[i] * fct;
    const auto res = std::sqrt(ri * ri + ii * ii);
    out[i] = res;
  }

#endif
}

// NOLINTEND(*-pointer-arithmetic, *-magic-numbers)
