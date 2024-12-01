#include "similiarity.hpp"
#include <fmt/format.h>
#include <vector>

int main() {
  std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> b = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    auto sim = cosine_similarity_naive(a.data(), b.data(), a.size());
    fmt::println("Similarity (naive): {}", sim);
  }

#if defined(__AVX2__)
  {
    auto sim = cosine_similarity_avx2(a.data(), b.data(), a.size());
    fmt::println("Similarity (avx2): {}", sim);
  }
#endif

#if defined(__ARM_NEON__)
  {
    auto sim = cosine_similarity_neon(a.data(), b.data(), a.size());
    fmt::println("Similarity (neon): {}", sim);
  }
#endif

  return 0;
}