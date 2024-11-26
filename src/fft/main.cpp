#include "fft_wrapper.hpp"
#include <cmath>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <vector>

int main() {
  fmt::println("Hello world");

#ifdef IPP_FOUND
  get_ipp_version();

  int N = 128;

  std::vector<float> src(N);
  std::vector<float> dst(N);

  for (int i = 0; i < N; ++i) {
    src[i] = sin(2 * IPP_PI * i / N);
  }

  FFT_1D_IPP engine(N);
  engine.forward_RToPack(src, dst);

  fmt::println("IPP FFT src: {}", fmt::join(src, ", "));
  fmt::println("IPP FFT Result: {}", fmt::join(dst, ", "));

  {

    // Set the size
    const int N = 128;

    // Spec and working buffers
    IppsDFTSpec_C_32fc *pDFTSpec = 0;
    Ipp8u *pDFTInitBuf, *pDFTWorkBuf;

    // Allocate complex buffers
    Ipp32fc *pSrc = ippsMalloc_32fc(N);
    Ipp32fc *pDst = ippsMalloc_32fc(N);

    // Query to get buffer sizes
    int sizeDFTSpec, sizeDFTInitBuf, sizeDFTWorkBuf;
    auto status =
        ippsDFTGetSize_C_32fc(N, IPP_FFT_NODIV_BY_ANY, ippAlgHintAccurate,
                              &sizeDFTSpec, &sizeDFTInitBuf, &sizeDFTWorkBuf);
    handleIppError(status);

    // Alloc DFT buffers
    pDFTSpec = (IppsDFTSpec_C_32fc *)ippsMalloc_8u(sizeDFTSpec);
    pDFTInitBuf = ippsMalloc_8u(sizeDFTInitBuf);
    pDFTWorkBuf = ippsMalloc_8u(sizeDFTWorkBuf);

    // Initialize DFT
    status = ippsDFTInit_C_32fc(N, IPP_FFT_NODIV_BY_ANY, ippAlgHintAccurate,
                                pDFTSpec, pDFTInitBuf);
    handleIppError(status);

    if (pDFTInitBuf != nullptr) {
      ippFree(pDFTInitBuf);
    }

    // Do the DFT
    status = ippsDFTFwd_CToC_32fc(pSrc, pDst, pDFTSpec, pDFTWorkBuf);
    handleIppError(status);

    // check results
    status = ippsDFTInv_CToC_32fc(pDst, pDst, pDFTSpec, pDFTWorkBuf);
    handleIppError(status);
    int OK = 1;
    for (int i = 0; i < N; i++) {
      pDst[i].re /= (Ipp32f)N;
      pDst[i].im /= (Ipp32f)N;
      if ((abs(pSrc[i].re - pDst[i].re) > .001) ||
          (abs(pSrc[i].im - pDst[i].im) > .001)) {
        OK = 0;
        break;
      }
    }
    puts(OK == 1 ? "DFT OK" : "DFT Fail");

    if (pDFTWorkBuf != nullptr) {
      ippFree(pDFTWorkBuf);
    }
    if (pDFTSpec != nullptr) {
      ippFree(pDFTSpec);
    }

    ippFree(pSrc);
    ippFree(pDst);
  }

#endif
}