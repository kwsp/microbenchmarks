#pragma once

#include "fftw.hpp"
#include <cassert>
#include <span>

/**
@brief Compute the analytic signal, using the Hilbert transform.
*/
template <fftw::Floating T>
void hilbert_abs(const std::span<const T> x, const std::span<T> env) {
  const auto n = x.size();
  assert(n > 0);
  assert(x.size() == env.size());

  auto &engine = fftw::engine_dft_1d<T>::get(n);

  // Copy input to real buffer
  // NOLINTBEGIN(*-pointer-arithmetic, *-magic-numbers)
  for (int i = 0; i < n; ++i) {
    engine.in[i][0] = x[i];
    engine.in[i][1] = 0.;
  }

  // Execute r2c fft
  engine.forward();

  // Zero negative frequencies (half-Hermitian to Hermitian conversion)
  // Double the magnitude of positive frequencies
  const auto n_half = n / 2;
  for (auto i = 1; i < n_half; ++i) {
    engine.out[i][0] *= 2.;
    engine.out[i][1] *= 2.;
  }

  if (n % 2 == 0) {
    engine.out[n_half][0] = 0.;
    engine.out[n_half][1] = 0.;
  } else {
    engine.out[n_half][0] *= 2.;
    engine.out[n_half][1] *= 2.;
  }

  for (auto i = n_half + 1; i < n; ++i) {
    engine.out[i][0] = 0.;
    engine.out[i][1] = 0.;
  }

  // Execute c2r fft on modified spectrum
  engine.backward();

  // Construct the analytic signal
  const T fct = static_cast<T>(1. / n);
  for (auto i = 0; i < n; ++i) {
    const auto real = x[i];
    const auto imag = engine.in[i][1] * fct;
    env[i] = std::abs(std::complex<T>{real, imag});
  }
  // NOLINTEND(*-pointer-arithmetic, *-magic-numbers)
}
