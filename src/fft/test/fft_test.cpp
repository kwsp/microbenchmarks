#include "../fft/Allocator.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(AllocatorTest, Default) {
  uspam::DefaultAllocator allocator;
  auto *p = allocator.allocate<int>(100);
  ASSERT_NE(p, nullptr);
  allocator.deallocate(p);
}

TEST(AllocatorTest, FFTW) {
  uspam::FFTW3Allocator allocator;
  auto *p = allocator.allocate<int>(100);
  ASSERT_NE(p, nullptr);
  allocator.deallocate(p);
}

TEST(AllocatorTest, StdAdaptor) {
  std::vector<int, uspam::StdAllocator<int>> vec1;
}