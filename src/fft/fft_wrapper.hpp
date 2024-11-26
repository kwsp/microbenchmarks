#pragma once

#include <fmt/format.h>
#include <iostream>
#include <span>

// Intel IPP
#ifdef IPP_FOUND

#include <ipp.h>
#include <ipp/ipps.h>

void handleIppError(IppStatus status) {
  if (status != ippStsNoErr) {
    std::cerr << "IPP Error: " << ippGetStatusString(status) << " (" << status
              << ")\n";
    exit(EXIT_FAILURE); // Or handle the error gracefully
  }
}

inline void get_ipp_version() {
  const auto *const lib = ippsGetLibVersion();
  fmt::println("\nIPP version\t{}.{}", lib->major, lib->minor);
  fmt::println("IPP majorBuild\t{}", lib->majorBuild);
  fmt::println("IPP build\t{}", lib->build);
  fmt::println("IPP targetCpu\t{}", lib->targetCpu);
  fmt::println("IPP Name\t{}", lib->Name);
  fmt::println("IPP Version\t{}", lib->Version);
  fmt::println("IPP BuildDate\t{}\n", lib->BuildDate);

  IppStatus status{};
  Ipp64u cpuFeatures{};
  Ipp64u enabledFeatures{};

  status = ippGetCpuFeatures(&cpuFeatures, nullptr);
  if (status != ippStsNoErr) {
    return;
  }
  enabledFeatures = ippGetEnabledCpuFeatures();

  const auto printInfo = [cpuFeatures, enabledFeatures](int feature,
                                                        const char *name) {
    fmt::println("{}\t{}\t{}", name, (feature & cpuFeatures) ? 'Y' : 'N',
                 (feature & enabledFeatures) ? 'Y' : 'N');
  };

  fmt::println("Features supported by CPU\tby Intel IPP");
  fmt::println("------------------------------------------");
  printInfo(ippCPUID_MMX, "MXX");
  printInfo(ippCPUID_SSE, "SSE");
  printInfo(ippCPUID_SSE2, "SSE2");
  printInfo(ippCPUID_SSE3, "SSE3");
  printInfo(ippCPUID_SSSE3, "SSSE3");
  printInfo(ippCPUID_MOVBE, "MOVBE");
  printInfo(ippCPUID_SSE41, "SSE41");
  printInfo(ippCPUID_SSE42, "SSE42");
  printInfo(ippCPUID_AVX, "AVX");
  printInfo(ippAVX_ENABLEDBYOS, "AVX enabled");
  printInfo(ippCPUID_AES, "AES");
  printInfo(ippCPUID_CLMUL, "CLMUL");
  printInfo(ippCPUID_RDRAND, "RDRAND");
  printInfo(ippCPUID_F16C, "F16C");
  printInfo(ippCPUID_AVX2, "AVX2");
  printInfo(ippCPUID_ADCOX, "ADCOX");
  printInfo(ippCPUID_RDSEED, "RDSEED");
  printInfo(ippCPUID_PREFETCHW, "PREFETCHW");
  printInfo(ippCPUID_SHA, "SHA");
  if (ippAVX512_ENABLEDBYOS & cpuFeatures) {
    printInfo(ippCPUID_AVX512F, "AVX512F");
    printInfo(ippCPUID_AVX512CD, "AVX512CD");
    printInfo(ippCPUID_AVX512ER, "AVX512ER");
    printInfo(ippCPUID_AVX512PF, "AVX512PF");
    printInfo(ippCPUID_AVX512BW, "AVX512BW");
    printInfo(ippCPUID_AVX512VL, "AVX512VL");
    printInfo(ippCPUID_AVX512VBMI, "AVX512VBMI");
    printInfo(ippCPUID_MPX, "MPX");
    printInfo(ippCPUID_AVX512_4FMADDPS, "AVX512_4FMADDPS");
    printInfo(ippCPUID_AVX512_4VNNIW, "AVX512_4VNNIW");
    printInfo(ippCPUID_KNC, "KNC");
    printInfo(ippCPUID_AVX512IFMA, "AVX512IFMA");
    printInfo(ippAVX512_ENABLEDBYOS, "AVX512 enabled");
  }

  fmt::println("\n");
}

class FFT_1D_IPP {
public:
  FFT_1D_IPP(const FFT_1D_IPP &) = delete;
  FFT_1D_IPP(FFT_1D_IPP &&) = delete;
  FFT_1D_IPP &operator=(const FFT_1D_IPP &) = delete;
  FFT_1D_IPP &operator=(FFT_1D_IPP &&) = delete;

  explicit FFT_1D_IPP(int size) {
    int flag = IPP_FFT_DIV_INV_BY_N;
    IppHintAlgorithm hint = ippAlgHintAccurate;
    IppStatus status{};

    int sizeSpec{};
    int sizeInitBuf{};
    int sizeWorkBuf{};
    // Get size of FFT specification and work buffers
    status = ippsDFTGetSize_R_32f(size, flag, hint, &sizeSpec, &sizeInitBuf,
                                  &sizeWorkBuf);
    handleIppError(status);

    // Allocate FFT spec and buffers
    specBuffer = {ippsMalloc_8u(sizeSpec), static_cast<size_t>(sizeSpec)};
    workBuffer = {ippsMalloc_8u(sizeWorkBuf), static_cast<size_t>(sizeWorkBuf)};

    Ipp8u *initBuffer = ippsMalloc_8u(sizeInitBuf);

    // Initialize FFT spec struct
    status = ippsDFTInit_R_32f(size, flag, hint, pDFTSpec, initBuffer);
    handleIppError(status);

    if (initBuffer != nullptr) {
      ippsFree(initBuffer);
    }
  }

  ~FFT_1D_IPP() {
    if (specBuffer.data() != nullptr) {
      ippsFree(specBuffer.data());
    }
    if (workBuffer.data() != nullptr) {
      ippsFree(workBuffer.data());
    }
  }

  void forward_RToPack(std::span<float> real, std::span<float> complex) {
    auto status = ippsDFTFwd_RToPack_32f(real.data(), complex.data(), pDFTSpec,
                                         workBuffer.data());
    handleIppError(status);
  }

  void backward_PackToR(std::span<float> complex, std::span<float> real) {
    auto status = ippsDFTInv_PackToR_32f(complex.data(), real.data(), pDFTSpec,
                                         workBuffer.data());
    handleIppError(status);
  }

private:
  IppsDFTSpec_R_32f *pDFTSpec{};

  std::span<Ipp8u> specBuffer;
  std::span<Ipp8u> workBuffer;
};

#endif
