#pragma once

#include "fftw.hpp"
#include <cassert>
#include <span>

template <typename T>
void hilbert_abs_cx_scale_imag(T const *real, T const *imag, T fct, size_t n,
                               T *out) {
  for (auto i = 0; i < n; ++i) {
    const auto ri = real[i];
    const auto ii = imag[i] * fct;
    const auto res = std::sqrt(real * real + imag * imag);
    out[i] = res;
  }
}

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
  // NOLINTBEGIN(*-pointer-arithmetic, *-magic-numbers)
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
  // NOLINTEND(*-pointer-arithmetic, *-magic-numbers)
}
