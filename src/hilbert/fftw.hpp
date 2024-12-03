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
  [[nodiscard]] static Plan FUNC(PARAMS) {                                     \
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
                  COMMA int odist COMMA unsigned flags,
      rank COMMA n COMMA howmany COMMA in COMMA inembed COMMA istride COMMA
          idist COMMA out COMMA onembed COMMA ostride COMMA odist COMMA flags)
  PLAN_CREATE_METHOD(
      many_dft_c2r,
      int rank COMMA const int *n COMMA int howmany COMMA Complex<T> *in
          COMMA const int *inembed COMMA int istride COMMA int idist COMMA
              T *out COMMA const int *onembed COMMA int ostride COMMA int odist
                  COMMA unsigned flags,
      rank COMMA n COMMA howmany COMMA in COMMA inembed COMMA istride COMMA
          idist COMMA out COMMA onembed COMMA ostride COMMA odist COMMA flags)

  /**
   * Advanced Real-to-real Transforms
   * https://fftw.org/fftw3_doc/Advanced-Real_002dto_002dreal-Transforms.html
   */
  PLAN_CREATE_METHOD(
      many_r2r,
      int rank COMMA const int *n COMMA int howmany COMMA T *in
          COMMA const int *inembed COMMA int istride COMMA int idist COMMA
              T *out COMMA const int *onembed COMMA int ostride COMMA int odist
                  COMMA R2RKind<T> *kind COMMA unsigned flags,
      rank COMMA n COMMA howmany COMMA in COMMA inembed COMMA istride COMMA
          idist COMMA out COMMA onembed COMMA ostride COMMA odist COMMA kind
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

template <typename T> struct C2CBuffer {
  using Cx = fftw::Complex<T>;
  Cx *in, *out;
  explicit C2CBuffer(size_t n)
      : in(fftw::alloc_complex<T>(n)), out(fftw::alloc_complex<T>(n)) {}
  C2CBuffer(const C2CBuffer &) = delete;
  C2CBuffer(C2CBuffer &&) = delete;
  C2CBuffer &operator=(const C2CBuffer &) = delete;
  C2CBuffer &operator=(C2CBuffer &&) = delete;
  ~C2CBuffer() noexcept {
    if (in) fftw::free<T>(in);
    if (out) fftw::free<T>(out);
  }
};

template <typename T> struct C2CSplitBuffer {
  using Cx = fftw::Complex<T>;
  T *ri, *ii, *ro, *io;
  explicit C2CSplitBuffer(size_t n)
      : ri(fftw::alloc_real<T>(n)), ii(fftw::alloc_real<T>(n)),
        ro(fftw::alloc_real<T>(n)), io(fftw::alloc_real<T>(n)) {}
  C2CSplitBuffer(const C2CSplitBuffer &) = delete;
  C2CSplitBuffer(C2CSplitBuffer &&) = delete;
  C2CSplitBuffer &operator=(const C2CSplitBuffer &) = delete;
  C2CSplitBuffer &operator=(C2CSplitBuffer &&) = delete;
  ~C2CSplitBuffer() noexcept {
    if (ri) fftw::free<T>(ri);
    if (ii) fftw::free<T>(ii);
    if (ro) fftw::free<T>(ro);
    if (io) fftw::free<T>(io);
  }
};

template <typename T> struct R2CBuffer {
  using Cx = fftw::Complex<T>;
  T *in;
  Cx *out;
  explicit R2CBuffer(size_t n)
      : in(fftw::alloc_real<T>(n)), out(fftw::alloc_complex<T>(n / 2 + 1)) {}
  R2CBuffer(const R2CBuffer &) = delete;
  R2CBuffer(R2CBuffer &&) = delete;
  R2CBuffer &operator=(const R2CBuffer &) = delete;
  R2CBuffer &operator=(R2CBuffer &&) = delete;
  ~R2CBuffer() noexcept {
    if (in) fftw::free<T>(in);
    if (out) fftw::free<T>(out);
  }
};

template <typename T> struct R2CSplitBuffer {
  using Cx = fftw::Complex<T>;
  T *in, *ro, *io;
  explicit R2CSplitBuffer(size_t n)
      : in(fftw::alloc_real<T>(n)), ro(fftw::alloc_real<T>(n / 2 + 1)),
        io(fftw::alloc_real<T>(n / 2 + 1)) {}
  R2CSplitBuffer(const R2CSplitBuffer &) = delete;
  R2CSplitBuffer(R2CSplitBuffer &&) = delete;
  R2CSplitBuffer &operator=(const R2CSplitBuffer &) = delete;
  R2CSplitBuffer &operator=(R2CSplitBuffer &&) = delete;
  ~R2CSplitBuffer() noexcept {
    if (in) fftw::free<T>(in);
    if (ro) fftw::free<T>(ro);
    if (io) fftw::free<T>(io);
  }
};

template <Floating T> struct EngineDFT1D : public cache_mixin<EngineDFT1D<T>> {
  using Cx = fftw::Complex<T>;
  using Plan = fftw::Plan<T>;

  C2CBuffer<T> buf;
  Plan plan_forward;
  Plan plan_backward;

  explicit EngineDFT1D(size_t n)
      : buf(n), plan_forward(Plan::dft_1d(n, buf.in, buf.out, FFTW_FORWARD,
                                          FFTW_ESTIMATE)),
        plan_backward(
            Plan::dft_1d(n, buf.out, buf.in, FFTW_BACKWARD, FFTW_ESTIMATE)){};

  void forward() { plan_forward.execute(); }
  void forward(Cx *in, Cx *out) const { plan_forward.execute(in, out); }

  void backward() { plan_backward.execute(); }
  void backward(Cx *in, Cx *out) const { plan_backward.execute(in, out); }
};

template <Floating T>
struct EngineDFTSplit1D : public cache_mixin<EngineDFTSplit1D<T>> {
  using Cx = fftw::Complex<T>;
  using Plan = fftw::Plan<T>;

  C2CSplitBuffer<T> buf;
  IODim<T> dim;
  Plan plan_forward;
  Plan plan_backward;

  explicit EngineDFTSplit1D(size_t n)
      : buf(n), dim(IODim<T>{.n = (int)n, .is = 1, .os = 1}),
        plan_forward(Plan::guru_split_dft(1, &dim, 0, nullptr, buf.ri, buf.ii,
                                          buf.ro, buf.io, FFTW_ESTIMATE)),
        plan_backward(Plan::guru_split_dft(1, &dim, 0, nullptr, buf.io, buf.ro,
                                           buf.ii, buf.ri, FFTW_ESTIMATE)){
            /*
            https://fftw.org/fftw3_doc/Guru-Complex-DFTs.html#Guru-Complex-DFTs
            There is no sign parameter in fftw_plan_guru_split_dft. This
            function always plans for an FFTW_FORWARD transform. To plan for an
            FFTW_BACKWARD transform, you can exploit the identity that the
            backwards DFT is equal to the forwards DFT with the real and
            imaginary parts swapped.
            */
        };

  void forward() { plan_forward.execute(); }
  void forward(Cx *in, Cx *out) const { plan_forward.execute(in, out); }

  void backward() { plan_backward.execute(); }
  void backward(Cx *in, Cx *out) const { plan_backward.execute(in, out); }
};

template <Floating T> struct EngineR2C1D : public cache_mixin<EngineR2C1D<T>> {
  using Cx = fftw::Complex<T>;
  using Plan = fftw::Plan<T>;

  R2CBuffer<T> buf;
  Plan plan_forward;
  Plan plan_backward;

  explicit EngineR2C1D(size_t n)
      : buf(n), plan_forward(Plan::dft_r2c_1d(static_cast<int>(n), buf.in,
                                              buf.out, FFTW_ESTIMATE)),
        plan_backward(Plan::dft_c2r_1d(static_cast<int>(n), buf.out, buf.in,
                                       FFTW_ESTIMATE)) {}

  void forward() const { plan_forward.execute(); }
  void forward(T *in, Cx *out) const { plan_forward.execute_dft_r2c(in, out); }

  void backward() const { plan_forward.execute(); }
  void backward(Cx *in, T *out) const { plan_forward.execute_dft_c2r(in, out); }
};

/**
Helper functions
 */

/**
out[i] += in[i] * fct
 */
template <typename t>
inline void normalize_add(t *out, t *in, size_t len, t fct) {
  for (size_t i = 0; i < len; ++i) {
    out[i] += in[i] * fct;
  }
}
/**
in[i] *= fct
 */
template <typename t> inline void normalize(t *in, size_t len, t fct) {
  for (size_t i = 0; i < len; ++i) {
    in[i] *= fct;
  }
}

} // namespace fftw
