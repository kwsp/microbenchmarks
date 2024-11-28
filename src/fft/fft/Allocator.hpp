#pragma once

#include <cstdlib>
#include <fftw3.h>

namespace uspam {

// Base class for custom allocators
class Allocator {
public:
  virtual ~Allocator() = default;

  template <typename T> T *allocate(size_t count) {
    return static_cast<T *>(allocateBytes(count * sizeof(T)));
  };
  virtual void *allocateBytes(size_t size) = 0;
  virtual void deallocate(void *ptr) = 0;
};

class DefaultAllocator final : public Allocator {
public:
  // NOLINTBEGIN(*-no-malloc)
  void *allocateBytes(size_t size) override { return std::malloc(size); }
  void deallocate(void *ptr) override { std::free(ptr); }
  // NOLINTEND(*-no-malloc)
};

class AllocatorManager {
public:
  static Allocator &getInstance() {
    thread_local DefaultAllocator defaultAllocator;
    return currentAllocator != nullptr ? *currentAllocator : defaultAllocator;
  }

  static void setAllocator(Allocator *allocator) {
    currentAllocator = allocator;
  }

private:
  static inline thread_local Allocator *currentAllocator{nullptr}; // NOLINT
};

// Adaptor for std containers
template <typename T> class StdAllocator {
public:
  using value_type = T;

  StdAllocator() noexcept : allocator(&AllocatorManager::getInstance()) {}

private:
  Allocator *allocator;

  // Allow other 'StdAllocator' types to access private members
  template <typename U> friend class StdAllocator;
};

class FFTW3Allocator final : public Allocator {
public:
  void *allocateBytes(size_t size) override { return fftw_malloc(size); }
  void deallocate(void *ptr) override { fftw_free(ptr); }
};

} // namespace uspam
