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

// Place this at the beginning of main() and RAII will take care of setting up
// and tearing down FFTW3 (threads and wisdom)
// NOLINTNEXTLINE(*-special-member-functions)
struct WisdomSetup {
  WisdomSetup() {
    static bool callSetup = true;
    if (callSetup) {
      fftw_make_planner_thread_safe();
      callSetup = false;
    }
    fftw_import_wisdom_from_filename(".fftw_wisdom");
    fftwf_import_wisdom_from_filename(".fftwf_wisdom");
  }
  ~WisdomSetup() {
    fftw_export_wisdom_to_filename(".fftw_wisdom");
    fftwf_export_wisdom_to_filename(".fftwf_wisdom");
  }
};

template <typename T>
concept Floating = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <Floating T> struct Traits {
  using Real = T;
  using Complex = std::complex<T>;
};
template <> struct Traits<double> {
  using FftwCx = fftw_complex;
  using PlanT = fftw_plan;
  using IODim = fftw_iodim;
  using R2RKind = fftw_r2r_kind;
  using IODim64 = fftw_iodim64;
};
template <> struct Traits<float> {
  using FftwCx = fftwf_complex;
  using PlanT = fftwf_plan;
  using IODim = fftwf_iodim;
  using R2RKind = fftwf_r2r_kind;
  using IODim64 = fftwf_iodim64;
};

template <Floating T> using Complex = Traits<T>::FftwCx;
template <Floating T> using PlanT = Traits<T>::PlanT;
template <Floating T> using IODim = Traits<T>::IODim;
template <Floating T> using R2RKind = Traits<T>::R2RKind;
template <Floating T> using IODim64 = Traits<T>::IODim64;

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
        static_assert(false, "Not supported");                                 \
      }                                                                        \
    }()};                                                                      \
    return planner;                                                            \
  }

#define PLAN_EXECUTE_METHOD(FUNC, PARAMS, PARAMS_CALL)                         \
  void FUNC(PARAMS) {                                                          \
    assert(plan != nullptr);                                                   \
    if constexpr (std::is_same_v<T, double>) {                                 \
      CONCAT(fftw_, FUNC)(plan, PARAMS_CALL);                                  \
    } else if constexpr (std::is_same_v<T, float>) {                           \
      CONCAT(fftwf_, FUNC)(plan, PARAMS_CALL);                                 \
    } else {                                                                   \
      static_assert(false, "Not supported");                                   \
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

  /**
   * Basic Interface
   */

  /**
   * Complex DFTs
   * https://fftw.org/fftw3_doc/Complex-DFTs.html
   */

  PLAN_CREATE_METHOD(dft_1d,
                     int n COMMA Complex<T> *in COMMA Complex<T> *out
                         COMMA int sign COMMA unsigned int flags,
                     n COMMA in COMMA out COMMA sign COMMA flags)
  PLAN_CREATE_METHOD(dft_2d,
                     int n0 COMMA int n1 COMMA Complex<T> *in COMMA Complex<T>
                         *out COMMA int sign COMMA unsigned int flags,
                     n0 COMMA n1 COMMA in COMMA out COMMA sign COMMA flags)
  PLAN_CREATE_METHOD(
      dft_3d,
      int n0 COMMA int n1 COMMA int n2 COMMA Complex<T> *in COMMA
          Complex<T> *out COMMA int sign COMMA unsigned int flags,
      n0 COMMA n1 COMMA n2 COMMA in COMMA out COMMA sign COMMA flags)
  PLAN_CREATE_METHOD(dft,
                     int rank COMMA int *n COMMA Complex<T> *in COMMA Complex<T>
                         *out COMMA int sign COMMA unsigned int flags,
                     rank COMMA n COMMA in COMMA out COMMA sign COMMA flags)

  /**
   * Real-data DFTs
   * https://fftw.org/fftw3_doc/Real_002ddata-DFTs.html
   */
  PLAN_CREATE_METHOD(dft_r2c_1d,
                     int n COMMA T *in COMMA Complex<T> *out
                         COMMA unsigned int flags,
                     n COMMA in COMMA out COMMA flags)
  PLAN_CREATE_METHOD(dft_r2c_2d,
                     int n0 COMMA int n1 COMMA T *in COMMA Complex<T> *out
                         COMMA unsigned int flags,
                     n0 COMMA n1 COMMA in COMMA out COMMA flags)
  PLAN_CREATE_METHOD(dft_r2c_3d,
                     int n0 COMMA int n1 COMMA int n2 COMMA T *in COMMA
                         Complex<T> *out COMMA unsigned int flags,
                     n0 COMMA n1 COMMA n2 COMMA in COMMA out COMMA flags)
  PLAN_CREATE_METHOD(dft_r2c,
                     int rank COMMA int *n COMMA T *in COMMA Complex<T> *out
                         COMMA unsigned int flags,
                     rank COMMA n COMMA in COMMA out COMMA flags)

  PLAN_CREATE_METHOD(dft_c2r_1d,
                     int n COMMA Complex<T> *in COMMA T *out
                         COMMA unsigned int flags,
                     n COMMA in COMMA out COMMA flags)
  PLAN_CREATE_METHOD(dft_c2r_2d,
                     int n0 COMMA int n1 COMMA Complex<T> *in COMMA T *out
                         COMMA unsigned int flags,
                     n0 COMMA n1 COMMA in COMMA out COMMA flags)
  PLAN_CREATE_METHOD(dft_c2r_3d,
                     int n0 COMMA int n1 COMMA int n2 COMMA Complex<T> *in COMMA
                         T *out COMMA unsigned int flags,
                     n0 COMMA n1 COMMA n2 COMMA in COMMA out COMMA flags)
  PLAN_CREATE_METHOD(dft_c2r,
                     int rank COMMA int *n COMMA Complex<T> *in COMMA T *out
                         COMMA unsigned int flags,
                     rank COMMA n COMMA in COMMA out COMMA flags)
  /**
   * Real-to-Real Transforms
   * https://fftw.org/fftw3_doc/Real_002dto_002dReal-Transforms.html
   */
  PLAN_CREATE_METHOD(r2r_1d,
                     int n COMMA T *in COMMA T *out COMMA R2RKind<T> kind
                         COMMA unsigned int flags,
                     n COMMA in COMMA out COMMA kind COMMA flags)
  PLAN_CREATE_METHOD(
      r2r_2d,
      int n0 COMMA int n1 COMMA T *in COMMA T *out COMMA R2RKind<T> kind0 COMMA
          R2RKind<T>
              kind1 COMMA unsigned int flags,
      n0 COMMA n1 COMMA in COMMA out COMMA kind0 COMMA kind1 COMMA flags)
  PLAN_CREATE_METHOD(r2r_3d,
                     int n0 COMMA int n1 COMMA int n2 COMMA T *in COMMA T *out
                         COMMA R2RKind<T>
                             kind0 COMMA R2RKind<T>
                                 kind1 COMMA R2RKind<T>
                                     kind2 COMMA unsigned int flags,
                     n0 COMMA n1 COMMA n2 COMMA in COMMA out COMMA kind0 COMMA
                         kind1 COMMA kind2 COMMA flags)
  PLAN_CREATE_METHOD(r2r,
                     int rank COMMA const int *n COMMA T *in COMMA T *out
                         COMMA const R2RKind<T> *kind COMMA unsigned int flags,
                     rank COMMA n COMMA in COMMA out COMMA kind COMMA flags)

  /**
   * Advanced Complex DFTs
   * https://fftw.org/fftw3_doc/Advanced-Complex-DFTs.html
   */
  PLAN_CREATE_METHOD(
      many_dft,
      int rank COMMA const int *n COMMA int howmany COMMA Complex<T> *in
          COMMA const int *inembed COMMA int istride COMMA int idist COMMA
              Complex<T> *out COMMA const int *onembed COMMA int ostride
                  COMMA int odist COMMA int sign COMMA unsigned flags,
      rank COMMA n COMMA howmany COMMA in COMMA inembed COMMA istride COMMA
          idist COMMA out COMMA onembed COMMA ostride COMMA odist COMMA sign
              COMMA flags)

  /**
   * Advanced Real-data DFTs
   * https://fftw.org/fftw3_doc/Advanced-Real_002ddata-DFTs.html
   */
  PLAN_CREATE_METHOD(
      many_dft_r2c,
      int rank COMMA const int *n COMMA int howmany COMMA T *in
          COMMA const int *inembed COMMA int istride COMMA int idist COMMA
              Complex<T> *out COMMA const int *onembed COMMA int ostride
                  COMMA int odist COMMA int sign COMMA unsigned flags,
      rank COMMA n COMMA howmany COMMA in COMMA inembed COMMA istride COMMA
          idist COMMA out COMMA onembed COMMA ostride COMMA odist COMMA sign
              COMMA flags)
  PLAN_CREATE_METHOD(
      many_dft_c2r,
      int rank COMMA const int *n COMMA int howmany COMMA Complex<T> *in
          COMMA const int *inembed COMMA int istride COMMA int idist COMMA
              T *out COMMA const int *onembed COMMA int ostride COMMA int odist
                  COMMA int sign COMMA unsigned flags,
      rank COMMA n COMMA howmany COMMA in COMMA inembed COMMA istride COMMA
          idist COMMA out COMMA onembed COMMA ostride COMMA odist COMMA sign
              COMMA flags)

  /**
   * Advanced Real-to-real Transforms
   * https://fftw.org/fftw3_doc/Advanced-Real_002dto_002dreal-Transforms.html
   */
  PLAN_CREATE_METHOD(
      many_dft_r2r,
      int rank COMMA const int *n COMMA int howmany COMMA T *in
          COMMA const int *inembed COMMA int istride COMMA int idist COMMA
              T *out COMMA const int *onembed COMMA int ostride COMMA int odist
                  COMMA int sign COMMA unsigned flags,
      rank COMMA n COMMA howmany COMMA in COMMA inembed COMMA istride COMMA
          idist COMMA out COMMA onembed COMMA ostride COMMA odist COMMA sign
              COMMA flags)

  /**
   * Guru Complex DFTs
   * https://fftw.org/fftw3_doc/Guru-Complex-DFTs.html
   */
  PLAN_CREATE_METHOD(guru_dft,
                     int rank COMMA const IODim<T> *dims COMMA int howmany_rank
                         COMMA const IODim<T> *howmany_dims COMMA Complex<T> *in
                             COMMA Complex<T> *out COMMA int sign
                                 COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA out COMMA sign COMMA flags)
  PLAN_CREATE_METHOD(
      guru64_dft,
      int rank COMMA const IODim64<T> *dims COMMA int howmany_rank
          COMMA const IODim64<T> *howmany_dims COMMA Complex<T> *in COMMA
              Complex<T> *out COMMA int sign COMMA unsigned flags,
      rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA in COMMA out
          COMMA sign COMMA flags)
  PLAN_CREATE_METHOD(guru_split_dft,
                     int rank COMMA const IODim<T> *dims COMMA int howmany_rank
                         COMMA const IODim<T> *howmany_dims COMMA T *ri COMMA
                             T *ii COMMA T *ro COMMA T *io COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         ri COMMA ii COMMA ro COMMA io COMMA flags)
  PLAN_CREATE_METHOD(
      guru64_split_dft,
      int rank COMMA const IODim64<T> *dims COMMA int howmany_rank
          COMMA const IODim64<T> *howmany_dims COMMA T *ri COMMA T *ii COMMA
              T *ro COMMA T *io COMMA unsigned flags,
      rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA ri COMMA ii
          COMMA ro COMMA io COMMA flags)

  /**
   * Guru Real-data DFTs
   * https://fftw.org/fftw3_doc/Guru-Real_002ddata-DFTs.html
   */
  PLAN_CREATE_METHOD(guru_dft_r2c,
                     int rank COMMA const IODim<T> *dims COMMA int howmany_rank
                         COMMA const IODim<T> *howmany_dims COMMA T *in COMMA
                             Complex<T> *out COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA out COMMA flags);
  PLAN_CREATE_METHOD(guru64_dft_r2c,
                     int rank COMMA const IODim64<T> *dims COMMA int
                         howmany_rank COMMA const IODim64<T> *howmany_dims COMMA
                             T *in COMMA Complex<T> *out COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA out COMMA flags);
  PLAN_CREATE_METHOD(guru_split_dft_r2c,
                     int rank COMMA const IODim<T> *dims COMMA int howmany_rank
                         COMMA const IODim<T> *howmany_dims COMMA T *in COMMA
                             T *ro COMMA T *io COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA ro COMMA io COMMA flags);
  PLAN_CREATE_METHOD(guru64_split_dft_r2c,
                     int rank COMMA const IODim64<T> *dims COMMA int
                         howmany_rank COMMA const IODim64<T> *howmany_dims COMMA
                             T *in COMMA T *ro COMMA T *io COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA ro COMMA io COMMA flags);
  PLAN_CREATE_METHOD(guru_dft_c2r,
                     int rank COMMA const IODim<T> *dims COMMA int howmany_rank
                         COMMA const IODim<T> *howmany_dims COMMA
                             fftw_complex *in COMMA T *out COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA out COMMA flags);
  PLAN_CREATE_METHOD(guru64_dft_c2r,
                     int rank COMMA const IODim64<T> *dims COMMA int
                         howmany_rank COMMA const IODim64<T> *howmany_dims COMMA
                             fftw_complex *in COMMA T *out COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA out COMMA flags);
  PLAN_CREATE_METHOD(guru_split_dft_c2r,
                     int rank COMMA const IODim<T> *dims COMMA int howmany_rank
                         COMMA const IODim<T> *howmany_dims COMMA T *ri COMMA
                             T *ii COMMA T *out COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         ri COMMA ii COMMA out COMMA flags);
  PLAN_CREATE_METHOD(guru64_split_dft_c2r,
                     int rank COMMA const IODim64<T> *dims
                         COMMA int howmany_rank
                             COMMA const IODim64<T> *howmany_dims COMMA T *ri
                                 COMMA T *ii COMMA T *out COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         ri COMMA ii COMMA out COMMA flags);

  /**
   * Guru Real-to-real Transforms
   */
  PLAN_CREATE_METHOD(guru_r2r,
                     int rank COMMA const IODim<T> *dims COMMA int howmany_rank
                         COMMA const IODim<T> *howmany_dims COMMA T *in COMMA
                             T *out COMMA const R2RKind<T> *kind
                                 COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA out COMMA kind COMMA flags)
  PLAN_CREATE_METHOD(
      guru64_r2r,
      int rank COMMA const IODim64<T> *dims COMMA int howmany_rank
          COMMA const IODim64<T> *howmany_dims COMMA T *in COMMA T *out
              COMMA const R2RKind<T> *kind COMMA unsigned flags,
      rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA in COMMA out
          COMMA kind COMMA flags)

  void execute() {
    if constexpr (std::is_same_v<T, double>) {
      fftw_execute(plan);
    } else if constexpr (std::is_same_v<T, float>) {
      fftwf_execute(plan);
    } else {
      static_assert(false, "Not supported");
    }
  }

  /**
   * Array execute interface
   * https://fftw.org/fftw3_doc/New_002darray-Execute-Functions.html#New_002darray-Execute-Functions
   */
  PLAN_EXECUTE_METHOD(execute_dft, Complex<T> *in COMMA Complex<T> *out,
                      in COMMA out);
  PLAN_EXECUTE_METHOD(execute_split_dft,
                      T *ri COMMA T *ii COMMA T *ro COMMA T *io,
                      ri COMMA ii COMMA ro COMMA io)
  PLAN_EXECUTE_METHOD(execute_dft_r2c, T *in COMMA Complex<T> *out,
                      in COMMA out)
  PLAN_EXECUTE_METHOD(execute_split_dft_r2c, T *in COMMA T *ro COMMA T *io,
                      in COMMA ro COMMA io);
  PLAN_EXECUTE_METHOD(execute_dft_c2r, Complex<T> *in COMMA T *out,
                      in COMMA out)
  PLAN_EXECUTE_METHOD(execute_split_dft_c2r, T *ri COMMA T *ii COMMA T *out,
                      ri COMMA ii COMMA out)
  PLAN_EXECUTE_METHOD(execute_r2r, T *in COMMA T *out, plan COMMA in COMMA out)
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
