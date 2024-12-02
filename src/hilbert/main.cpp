#include "hilbert.hpp"
#include <Eigen/Dense>
#include <iostream>

#ifdef CONV1D_HAS_IPP
#include <ipp.h>
#endif

// NOLINTBEGIN(*-magic-numbers)

template <typename T> std::span<T> to_span(Eigen::VectorX<T> &vec) {
  return {vec.data(), static_cast<size_t>(vec.size())};
}

template <typename T>
std::span<const T> to_cspan(const Eigen::VectorX<T> &vec) {
  return {vec.data(), static_cast<size_t>(vec.size())};
}

int main(int argc, char *argv[]) {
  using T = double;
  constexpr int N = 10;

  Eigen::VectorX<T> inp(N);
  inp.setRandom();

  {
    Eigen::VectorX<T> out(N);
    hilbert_abs<T>(to_cspan(inp), to_span(out));
    std::cout << "inp: " << inp << "\n";
    std::cout << "hilbert_abs out: " << out << "\n";
  }
}

// NOLINTEND(*-magic-numbers)