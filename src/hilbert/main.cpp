#include "aligned_vector.hpp"
#include "hilbert.hpp"
#include <cmath>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <numbers>
#include <opencv2/core/base.hpp>

// NOLINTBEGIN(*-magic-numbers)

int main(int argc, char *argv[]) {
  using T = float;
  constexpr int N = 10;

  AlignedVector<T> in(N);
  for (int i = 0; i < N; ++i) {
    in[i] = std::cos(std::numbers::pi_v<T> * 4 * i / (N - 1));
  }

  {
    AlignedVector<T> out(N);
    hilbert_fftw<T>(in, out);

    fmt::println("=== hilbert_fftw ===");
    fmt::println("In: {}", fmt::join(in, ", "));
    fmt::println("Out: {}", fmt::join(out, ", "));
  }

  {
    AlignedVector<T> out(N);
    hilbert_fftw_split<T>(in, out);

    fmt::println("=== hilbert_fftw_split ===");
    fmt::println("In: {}", fmt::join(in, ", "));
    fmt::println("Out: {}", fmt::join(out, ", "));
  }

#if defined(HAS_IPP)

  {
    AlignedVector<T> out(N);
    hilbert_ipp<T>(in, out);

    fmt::println("=== IPP ===");
    fmt::println("In: {}", fmt::join(in, ", "));
    fmt::println("Out: {}", fmt::join(out, ", "));
  }

#endif
}

// NOLINTEND(*-magic-numbers)