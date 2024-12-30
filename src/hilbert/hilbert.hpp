#pragma once

#include "fftw.hpp"
#include <cassert>
#include <iostream>
#include <span>

// NOLINTBEGIN(*-pointer-arithmetic, *-magic-numbers)

/**
@brief Compute the analytic signal, using the Hilbert transform.
*/
template <fftw::Floating T>
void hilbert_fftw(const std::span<const T> x, const std::span<T> env) {
  const auto n = x.size();
  assert(n > 0);
  assert(x.size() == env.size());

  auto &engine = fftw::EngineDFT1D<T, true>::get(n);
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

  fftw::scale_and_magnitude<T>(buf.in, env.data(), n, fct);
}

/**
@brief Compute the analytic signal, using the Hilbert transform.
*/
template <fftw::Floating T>
void hilbert_fftw_r2c(const std::span<const T> x, const std::span<T> env) {
  const auto n = x.size();
  assert(n > 0);
  assert(x.size() == env.size());

  fftw::EngineR2C1D<T> &engine = fftw::EngineR2C1D<T>::get(n);
  fftw::R2CBuffer<T> &buf = engine.buf;

  // // Copy input to real buffer
  // for (int i = 0; i < n; ++i) {
  //   buf.in[i] = x[i];
  // }

  // // Execute r2c fft
  // engine.forward();

  // Avoid a copy
  // NOLINTNEXTLINE(*-const-cast)
  engine.forward(x.data(), buf.out);

  //  Multiply by 1j
  const auto cx_size = n / 2 + 1;
  for (auto i = 0; i < cx_size; ++i) {
    const auto re = buf.out[i][0];
    const auto im = buf.out[i][1];
    buf.out[i][0] = im;
    buf.out[i][1] = -re;
  }

  // Execute c2r fft on modified spectrum
  engine.backward();

  // Take the abs of the analytic signal
  const T fct = static_cast<T>(1. / n);

  for (auto i = 0; i < n; ++i) {
    const auto real = x[i];
    const auto imag = buf.in[i] * fct;
    env[i] = std::sqrt(real * real + imag * imag);
  }

  // fftw::scale_and_magnitude<T>(buf.in, env.data(), n, fct);
}
/**
@brief Compute the analytic signal, using the Hilbert transform.
*/
template <fftw::Floating T>
void hilbert_fftw_split(const std::span<const T> x, const std::span<T> env) {
  const size_t n = x.size();
  assert(n > 0);
  assert(x.size() == env.size());

  auto &engine = fftw::EngineDFTSplit1D<T, true>::get(n);
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
  fftw::scale_imag_and_magnitude(x.data(), buf.ii, fct, n, env.data());
}

#if defined(HAS_IPP)

#include <format>
#include <ipp.h>

void handleIppStatus(IppStatus status) {
  if (status != ippStsNoErr) {
    throw std::runtime_error(std::format("Ipp error: {}", status));
  }
}

namespace detail {

// In memory cache with key type K and value type V
// additionally accepts a mutex to guard the V constructor
template <class Key, class Val> auto get_cached(Key key) -> Val * {
  thread_local std::unordered_map<Key, std::unique_ptr<Val>> cache;

  auto &val = cache[key];
  if (val == nullptr) { val = std::make_unique<Val>(key); }
  return val.get();
}

struct HilbertIppBuf32fc {
  Ipp32fc *y;
  IppsHilbertSpec *pSpec;
  Ipp8u *pBuffer;

  HilbertIppBuf32fc() = delete;
  HilbertIppBuf32fc(const HilbertIppBuf32fc &) = delete;
  HilbertIppBuf32fc(HilbertIppBuf32fc &&) = delete;
  HilbertIppBuf32fc &operator=(const HilbertIppBuf32fc &) = delete;
  HilbertIppBuf32fc &operator=(HilbertIppBuf32fc &&) = delete;
  explicit HilbertIppBuf32fc(size_t n)
      : y(static_cast<Ipp32fc *>(
            ippMalloc(static_cast<int>(n * sizeof(Ipp32fc))))) {
    IppStatus status;
    int sizeSpec, sizeBuf;

    status = ippsHilbertGetSize_32f32fc(n, ippAlgHintNone, &sizeSpec, &sizeBuf);
    pSpec = static_cast<IppsHilbertSpec *>(ippMalloc(sizeSpec));
    pBuffer = static_cast<Ipp8u *>(ippMalloc(sizeBuf));
    status = ippsHilbertInit_32f32fc(n, ippAlgHintNone, pSpec, pBuffer);

    // TODO: handle status
  }
  ~HilbertIppBuf32fc() {
    ippFree(pSpec);
    ippFree(pBuffer);
    ippFree(y);
  }
};

struct HilbertIppBuf64fc {
  Ipp64fc *y;
  IppsHilbertSpec *pSpec;
  Ipp8u *pBuffer;

  HilbertIppBuf64fc() = delete;
  HilbertIppBuf64fc(const HilbertIppBuf64fc &) = delete;
  HilbertIppBuf64fc(HilbertIppBuf64fc &&) = default;
  HilbertIppBuf64fc &operator=(const HilbertIppBuf64fc &) = delete;
  HilbertIppBuf64fc &operator=(HilbertIppBuf64fc &&) = default;
  explicit HilbertIppBuf64fc(size_t n)
      : y(static_cast<Ipp64fc *>(
            ippMalloc(static_cast<int>(n * sizeof(Ipp64fc))))) {
    IppStatus status;
    int sizeSpec, sizeBuf;

    status = ippsHilbertGetSize_64f64fc(n, ippAlgHintNone, &sizeSpec, &sizeBuf);
    pSpec = static_cast<IppsHilbertSpec *>(ippMalloc(sizeSpec));
    pBuffer = static_cast<Ipp8u *>(ippMalloc(sizeBuf));
    status = ippsHilbertInit_64f64fc(n, ippAlgHintNone, pSpec, pBuffer);

    // TODO: handle status
  }
  ~HilbertIppBuf64fc() {
    ippFree(pSpec);
    ippFree(pBuffer);
    ippFree(y);
  }
};

} // namespace detail

template <fftw::Floating T>
void hilbert_ipp(const std::span<const T> x, const std::span<T> env) {
  const size_t n = x.size();
  IppStatus status;

  if constexpr (std::is_same_v<T, float>) {
    auto &buf = *fftw::get_cached<size_t, detail::HilbertIppBuf32fc>(n);

    status = ippsHilbert_32f32fc(x.data(), buf.y, buf.pSpec, buf.pBuffer);
    ippsMagnitude_32fc(buf.y, env.data(), n);

  } else if constexpr (std::is_same_v<T, double>) {
    auto &buf = *fftw::get_cached<size_t, detail::HilbertIppBuf64fc>(n);

    status = ippsHilbert_64f64fc(x.data(), buf.y, buf.pSpec, buf.pBuffer);
    ippsMagnitude_64fc(buf.y, env.data(), n);

  } else {
    static_assert(false, "Not supported.");
  }
}

#endif

#if defined(__APPLE__)

#include <Accelerate/Accelerate.h>

void hilbert_vDSP() {

  const vDSP_Length n = 8;
  vDSP_DFT_SetupD forward =
      vDSP_DFT_zrop_CreateSetupD(NULL, n, vDSP_DFT_FORWARD);
  vDSP_DFT_SetupD inverse =
      vDSP_DFT_zop_CreateSetupD(forward, n, vDSP_DFT_INVERSE);
  //  Look like a typo?  The real-to-complex DFT takes its input separated into
  //  the even- and odd-indexed elements.  Since the real signal is [ 1, 2, 3,
  //  ... ], signal[0] is 1, signal[2] is 3, and so on for the even indices.
  double even[n / 2] = {1, 3, 5, 7};
  double odd[n / 2] = {2, 4, 6, 8};
  double real[n] = {0};
  double imag[n] = {0};
  vDSP_DFT_ExecuteD(forward, even, odd, real, imag);
  //  At this point, we have the forward real-to-complex DFT, which agrees with
  //  MATLAB up to a factor of two.  Since we want to double all but DC and NY
  //  as part of the Hilbert transform anyway, I'm not going to bother to
  //  unscale the rest of the frequencies -- they're already the values that
  //  we really want.  So we just need to move NY into the "right place",
  //  and scale DC and NY by 0.5.  The reflection frequencies are already
  //  zeroed out because the real-to-complex DFT only writes to the first n/2
  //  elements of real and imag.
  real[0] *= 0.5;
  real[n / 2] = 0.5 * imag[0];
  imag[0] = 0.0;
  printf("Stage 2:\n");
  for (int i = 0; i < n; ++i)
    printf("%f%+fi\n", real[i], imag[i]);

  double hilbert[2 * n];
  double *hilbertreal = &hilbert[0];
  double *hilbertimag = &hilbert[n];
  vDSP_DFT_ExecuteD(inverse, real, imag, hilbertreal, hilbertimag);
  //  Now we have the completed hilbert transform up to a scale factor of n.
  //  We can unscale using vDSP_vsmulD.
  double scale = 1.0 / n;
  vDSP_vsmulD(hilbert, 1, &scale, hilbert, 1, 2 * n);
  printf("Stage 3:\n");
  for (int i = 0; i < n; ++i)
    printf("%f%+fi\n", hilbertreal[i], hilbertimag[i]);
  vDSP_DFT_DestroySetupD(inverse);
  vDSP_DFT_DestroySetupD(forward);
}

#endif

// NOLINTEND(*-pointer-arithmetic, *-magic-numbers)
