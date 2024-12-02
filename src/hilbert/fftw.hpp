/**
A C++ FFTW wrapper
 */
#pragma once

#include <complex>
#include <cstdlib>
#include <fftw3.h>
#include <span>
#include <type_traits>
#include <unordered_map>

namespace fftw {

template <typename T>
concept Floating = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <Floating T> struct Traits {
  using Real = T;
  using Complex = std::complex<T>;
};
template <> struct Traits<double> {
  using FftwCx = fftw_complex;
  using Plan = fftw_plan;
};
template <> struct Traits<float> {
  using FftwCx = fftwf_complex;
  using Plan = fftwf_plan;
};

template <Floating T> using Complex = Traits<T>::FftwCx;
template <Floating T> using PlanT = Traits<T>::Plan;

template <Floating T> struct prefix_;
template <> struct prefix_<double> {
  static constexpr const char *value = "fftw_";
};
template <> struct prefix_<float> {
  static constexpr const char *value = "fftwf_";
};
template <Floating T> inline constexpr const char *prefix = prefix_<T>::value;

// Macros to concatinate prefix to identifier
#define CONCAT(prefix, name) prefix##name
#define COMMA ,
#define TEMPLATIZE(ReturnT, FUNC, PARAMS, PARAMS_CALL)                         \
  template <Floating T> ReturnT FUNC(PARAMS) {                                 \
    if constexpr (std::is_same_v<T, double>) {                                 \
      return CONCAT(fftw_, FUNC)(PARAMS_CALL);                                 \
    } else if constexpr (std::is_same_v<T, float>) {                           \
      return CONCAT(fftwf_, FUNC)(PARAMS_CALL);                                \
    } else {                                                                   \
      static_assert(false, "Not implemented");                                 \
    }                                                                          \
  }

TEMPLATIZE(void *, malloc, size_t n, n)
TEMPLATIZE(T *, alloc_real, size_t n, n)
TEMPLATIZE(Complex<T> *, alloc_complex, size_t n, n)
TEMPLATIZE(void, free, void *n, n)

TEMPLATIZE(void, destroy_plan, PlanT<T> plan, plan)

#define PLAN_CREATE_METHOD(FUNC, PARAMS, PARAMS_CALL)                          \
  static Plan FUNC(PARAMS) {                                                   \
    Plan<T> planner{[&]() {                                                    \
      if constexpr (std::is_same_v<T, double>) {                               \
        return CONCAT(fftw_plan_, FUNC)(PARAMS_CALL);                          \
      } else if constexpr (std::is_same_v<T, float>) {                         \
        return CONCAT(fftwf_plan_, FUNC)(PARAMS_CALL);                         \
      } else {                                                                 \
        static_assert(false, "Not implemented");                               \
      }                                                                        \
    }()};                                                                      \
    return planner;                                                            \
  }

#define PLAN_EXECUTE_METHOD(FUNC, PARAMS, PARAMS_CALL)                         \
  void FUNC(PARAMS) {                                                          \
    if constexpr (std::is_same_v<T, double>) {                                 \
      CONCAT(fftw_, FUNC)(PARAMS_CALL);                                        \
    } else if constexpr (std::is_same_v<T, float>) {                           \
      CONCAT(fftwf_, FUNC)(PARAMS_CALL);                                       \
    } else {                                                                   \
      static_assert(false, "Not implemented");                                 \
    }                                                                          \
  }

template <typename T> struct Plan {
  PlanT<T> plan;

  Plan() = default;
  Plan(const Plan &) = delete;
  Plan(Plan &&) = default;
  Plan &operator=(const Plan &) = delete;
  Plan &operator=(Plan &&) = default;
  explicit Plan(PlanT<T> plan) : plan(std::move(plan)) {}
  ~Plan() {
    if (plan) { destroy_plan<T>(plan); }
  }

  PLAN_CREATE_METHOD(dft_1d,
                     int n COMMA Complex<T> *in COMMA Complex<T> *out
                         COMMA int sign COMMA unsigned int flags,
                     n COMMA in COMMA out COMMA sign COMMA flags)

  PLAN_CREATE_METHOD(dft_r2c_1d,
                     int n COMMA T *in COMMA Complex<T> *out
                         COMMA unsigned int flags,
                     n COMMA in COMMA out COMMA flags)

  PLAN_CREATE_METHOD(dft_c2r_1d,
                     int n COMMA Complex<T> *in COMMA T *out
                         COMMA unsigned int flags,
                     n COMMA in COMMA out COMMA flags)

  PLAN_EXECUTE_METHOD(execute, , plan)

  PLAN_EXECUTE_METHOD(execute_dft, Complex<T> *in COMMA Complex<T> *out,
                      plan COMMA in COMMA out);
  PLAN_EXECUTE_METHOD(execute_split_dft,
                      T *ri COMMA T *ii COMMA T *ro COMMA T *io,
                      plan COMMA ri COMMA ii COMMA ro COMMA io)
  PLAN_EXECUTE_METHOD(execute_r2r, T *in COMMA T *out, plan COMMA in COMMA out)
  PLAN_EXECUTE_METHOD(execute_dft_r2c, T *in COMMA Complex<T> *out,
                      plan COMMA in COMMA out)
  PLAN_EXECUTE_METHOD(execute_dft_c2r, Complex<T> *in COMMA T *out,
                      plan COMMA in COMMA out)
};

// In memory cache with key type K and value type V
// additionally accepts a mutex to guard the V constructor
template <class Key, class Val> auto get_cached(Key key) -> Val * {
  thread_local std::unordered_map<Key, std::unique_ptr<Val>> cache;

  auto &val = cache[key];
  if (val == nullptr) { val = std::make_unique<Val>(key); }
  return val.get();
}

template <typename Child> struct cache_mixin {
  static auto get(size_t n) -> Child & { return *get_cached<size_t, Child>(n); }
};

template <Floating T>
struct engine_dft_1d : public cache_mixin<engine_dft_1d<T>> {
  using Cx = fftw::Complex<T>;
  using Plan = fftw::Plan<T>;

  std::span<Cx> in;
  std::span<Cx> out;
  Plan plan_forward;
  Plan plan_backward;

  engine_dft_1d(const engine_dft_1d &) = delete;
  engine_dft_1d(engine_dft_1d &&) = delete;
  engine_dft_1d &operator=(const engine_dft_1d &) = delete;
  engine_dft_1d &operator=(engine_dft_1d &&) = delete;

  explicit engine_dft_1d(size_t n)
      : in(fftw::alloc_complex<T>(n), n), out(fftw::alloc_complex<T>(n), n),
        plan_forward(Plan::dft_1d(n, in.data(), out.data(), FFTW_FORWARD,
                                  FFTW_ESTIMATE)),
        plan_backward(Plan::dft_1d(n, out.data(), in.data(), FFTW_BACKWARD,
                                   FFTW_ESTIMATE)){};
  ~engine_dft_1d() {
    fftw::free<T>(in.data());
    fftw::free<T>(out.data());
  }

  void forward() { plan_forward.execute(); }
  void forward(Cx *inp, Cx *out) { plan_forward.execute(inp, out); }

  void backward() { plan_backward.execute(); }
  void backward(Cx *inp, Cx *out) { plan_backward.execute(inp, out); }
};

template <Floating T>
struct engine_r2c_1d : public cache_mixin<engine_r2c_1d<T>> {
  using Cx = fftw::Complex<T>;
  using Plan = fftw::Plan<T>;

  std::span<T> real;
  std::span<Cx> cx;
  Plan plan_forward;
  Plan plan_backward;

  engine_r2c_1d(const engine_r2c_1d &) = delete;
  engine_r2c_1d(engine_r2c_1d &&) = delete;
  engine_r2c_1d &operator=(const engine_r2c_1d &) = delete;
  engine_r2c_1d &operator=(engine_r2c_1d &&) = delete;

  explicit engine_r2c_1d(size_t n)
      : real(fftw::alloc_real<T>(n), n),
        cx(fftw::alloc_complex<T>(n / 2 + 1), n / 2 + 1),
        plan_forward(Plan::dft_r2c_1d(static_cast<int>(n), real.data(),
                                      cx.data(), FFTW_ESTIMATE)),
        plan_backward(
            Plan::dft_c2r_1d(static_cast<int>(n), cx.data(), FFTW_ESTIMATE)) {}

  ~engine_r2c_1d() {
    fftw::free<T>(real.data());
    fftw::free<T>(cx.data());
  }

  inline void forward() const { plan_forward.execute(); }
  inline void forward(T *inp, Cx *out) const {
    plan_forward.execute_dft_r2c(inp, out);
  }

  inline void backward() const { plan_forward.execute(); }
  inline void backward(Cx *inp, T *out) const {
    plan_forward.execute_dft_c2r(inp, out);
  }
};

} // namespace fftw
