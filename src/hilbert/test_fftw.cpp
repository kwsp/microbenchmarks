#include "fftw.hpp"
#include "hilbert.hpp"
#include <array>
#include <gtest/gtest.h>

class FFTWPlanCreateC2C : public testing::Test {
protected:
  using T = double;
  fftw::Complex<T> *in;
  fftw::Complex<T> *out;
  int n = 20;

  FFTWPlanCreateC2C() {
    in = fftw::alloc_complex<T>(n);
    out = fftw::alloc_complex<T>(n);
  }

  ~FFTWPlanCreateC2C() override {
    fftw::free<T>(in);
    fftw::free<T>(out);
  }
};

class FFTWPlanCreateR2C : public testing::Test {
protected:
  using T = double;
  T *in;
  fftw::Complex<T> *out;
  int n = 20;

  FFTWPlanCreateR2C() {
    in = fftw::alloc_real<T>(n);
    out = fftw::alloc_complex<T>(n / 2 + 1);
  }

  ~FFTWPlanCreateR2C() override {
    fftw::free<T>(in);
    fftw::free<T>(out);
  }
};

class FFTWPlanCreateR2R : public testing::Test {
protected:
  using T = double;
  T *in;
  T *out;
  int n = 20;

  FFTWPlanCreateR2R() {
    in = fftw::alloc_real<T>(n);
    out = fftw::alloc_real<T>(n);
  }

  ~FFTWPlanCreateR2R() override {
    fftw::free<T>(in);
    fftw::free<T>(out);
  }
};

class FFTWPlanCreateR2CSplit : public testing::Test {
protected:
  using T = double;
  T *in;
  T *ro;
  T *io;
  int n = 20;

  FFTWPlanCreateR2CSplit() {
    in = fftw::alloc_real<T>(n);
    ro = fftw::alloc_real<T>(n / 2 + 1);
    io = fftw::alloc_real<T>(n / 2 + 1);
  }

  ~FFTWPlanCreateR2CSplit() override {
    fftw::free<T>(in);
    fftw::free<T>(ro);
    fftw::free<T>(io);
  }
};

TEST_F(FFTWPlanCreateC2C, BasicPlan1d) {
  auto plan = fftw::Plan<T>::dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};
TEST_F(FFTWPlanCreateC2C, BasicPlan2d) {
  auto plan = fftw::Plan<T>::dft_2d(1, n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};
TEST_F(FFTWPlanCreateC2C, BasicPlan3d) {
  auto plan =
      fftw::Plan<T>::dft_3d(1, 1, n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};
TEST_F(FFTWPlanCreateC2C, BasicPlan) {
  int rank = 1;
  int dim = n;
  auto plan =
      fftw::Plan<T>::dft(rank, &dim, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};

TEST_F(FFTWPlanCreateR2C, BasicPlan1d) {
  auto plan_forward = fftw::Plan<T>::dft_r2c_1d(n, in, out, FFTW_ESTIMATE);
  ASSERT_NE(plan_forward.plan, nullptr);
  auto plan_backward = fftw::Plan<T>::dft_c2r_1d(n, out, in, FFTW_ESTIMATE);
  ASSERT_NE(plan_backward.plan, nullptr);
};
TEST_F(FFTWPlanCreateR2C, BasicPlan2d) {
  auto plan_forward = fftw::Plan<T>::dft_r2c_2d(1, n, in, out, FFTW_ESTIMATE);
  ASSERT_NE(plan_forward.plan, nullptr);
  auto plan_backward = fftw::Plan<T>::dft_c2r_2d(1, n, out, in, FFTW_ESTIMATE);
  ASSERT_NE(plan_backward.plan, nullptr);
};
TEST_F(FFTWPlanCreateR2C, BasicPlan3d) {
  auto plan_forward =
      fftw::Plan<T>::dft_r2c_3d(1, 1, n, in, out, FFTW_ESTIMATE);
  ASSERT_NE(plan_forward.plan, nullptr);
  auto plan_backward =
      fftw::Plan<T>::dft_c2r_3d(1, 1, n, out, in, FFTW_ESTIMATE);
  ASSERT_NE(plan_backward.plan, nullptr);
};
TEST_F(FFTWPlanCreateR2C, BasicPlan) {
  int rank = 1;
  int dim = n;
  auto planf = fftw::Plan<T>::dft_r2c(rank, &dim, in, out, FFTW_ESTIMATE);
  ASSERT_NE(planf.plan, nullptr);
  auto planb = fftw::Plan<T>::dft_c2r(rank, &dim, out, in, FFTW_ESTIMATE);
  ASSERT_NE(planb.plan, nullptr);
};

TEST_F(FFTWPlanCreateR2R, BasicPlan1d) {
  auto plan = fftw::Plan<T>::r2r_1d(n, in, out, FFTW_DHT, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};
TEST_F(FFTWPlanCreateR2R, BasicPlan2d) {
  auto plan =
      fftw::Plan<T>::r2r_2d(1, n, in, out, FFTW_DHT, FFTW_DHT, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};
TEST_F(FFTWPlanCreateR2R, BasicPlan3d) {
  auto plan = fftw::Plan<T>::r2r_3d(1, 1, n, in, out, FFTW_DHT, FFTW_DHT,
                                    FFTW_DHT, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};
TEST_F(FFTWPlanCreateR2R, BasicPlan) {
  int rank = 1;
  int dim = n;
  auto kind = FFTW_DHT;
  auto plan = fftw::Plan<T>::r2r(rank, &dim, in, out, &kind, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};

/*
Test advanced interface
*/
TEST_F(FFTWPlanCreateC2C, AdvancedPlan) {
  int rank = 1;
  int dim = n;
  int howmany = 1;
  int *inembed = &dim;
  int *onembed = &dim;
  auto pf = fftw::Plan<T>::many_dft(rank, &dim, howmany, in, inembed, 1, 0, out,
                                    onembed, 1, 0, FFTW_FORWARD, FFTW_ESTIMATE);
  ASSERT_NE(pf.plan, nullptr);
}

/*
Test advanced interface
*/
TEST_F(FFTWPlanCreateR2C, AdvancedPlan) {
  int rank = 1;
  int dim = n;
  int howmany = 1;
  int *inembed = &dim;
  int *onembed = &dim;
  auto pf = fftw::Plan<T>::many_dft_r2c(rank, &dim, howmany, in, inembed, 1, 0,
                                        out, onembed, 1, 0, FFTW_ESTIMATE);
  ASSERT_NE(pf.plan, nullptr);

  auto pb = fftw::Plan<T>::many_dft_c2r(rank, &dim, howmany, out, onembed, 1, 0,
                                        in, inembed, 1, 0, FFTW_ESTIMATE);
  ASSERT_NE(pb.plan, nullptr);
}
/*
Test advanced interface
*/
TEST_F(FFTWPlanCreateR2R, AdvancedPlan) {
  int rank = 1;
  int dim = n;
  int howmany = 1;
  int *inembed = &dim;
  int *onembed = &dim;
  auto kind = FFTW_DHT;
  auto pf = fftw::Plan<T>::many_r2r(rank, &dim, howmany, in, inembed, 1, 0, out,
                                    onembed, 1, 0, &kind, FFTW_ESTIMATE);
  ASSERT_NE(pf.plan, nullptr);
}

/*
Test guru interface
*/
TEST_F(FFTWPlanCreateC2C, GuruPlan) {
  {
    int rank = 1;
    int howmany = 1;
    fftw::IODim<T> dim{.n = n, .is = n, .os = n};
    fftw::IODim<T> howmany_dim{.n = n, .is = n, .os = n};

    auto pf = fftw::Plan<T>::guru_dft(rank, &dim, howmany, &howmany_dim, in,
                                      out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(pf.plan, nullptr);
  }
  {
    int rank = 1;
    int howmany = 1;
    fftw::IODim64<T> dim{.n = n, .is = n, .os = n};
    fftw::IODim64<T> howmany_dim{.n = n, .is = n, .os = n};

    auto pf = fftw::Plan<T>::guru64_dft(rank, &dim, howmany, &howmany_dim, in,
                                        out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(pf.plan, nullptr);
  }
}

TEST_F(FFTWPlanCreateR2C, GuruPlan) {
  {
    int rank = 1;
    int howmany = 1;
    fftw::IODim<T> dim{.n = n, .is = n, .os = n};
    fftw::IODim<T> howmany_dim{.n = n, .is = n, .os = n};

    auto pf = fftw::Plan<T>::guru_dft_r2c(rank, &dim, howmany, &howmany_dim, in,
                                          out, FFTW_ESTIMATE);
    ASSERT_NE(pf.plan, nullptr);

    auto pb = fftw::Plan<T>::guru_dft_c2r(rank, &dim, howmany, &howmany_dim,
                                          out, in, FFTW_ESTIMATE);
    ASSERT_NE(pb.plan, nullptr);
  }
  {
    int rank = 1;
    int howmany = 1;
    fftw::IODim64<T> dim{.n = n, .is = n, .os = n};
    fftw::IODim64<T> howmany_dim{.n = n, .is = n, .os = n};

    auto pf = fftw::Plan<T>::guru64_dft_r2c(rank, &dim, howmany, &howmany_dim,
                                            in, out, FFTW_ESTIMATE);
    ASSERT_NE(pf.plan, nullptr);

    auto pb = fftw::Plan<T>::guru64_dft_c2r(rank, &dim, howmany, &howmany_dim,
                                            out, in, FFTW_ESTIMATE);
    ASSERT_NE(pb.plan, nullptr);
  }
}

TEST_F(FFTWPlanCreateR2CSplit, GuruPlanSplit) {
  int rank = 1;
  fftw::IODim<T> dim{.n = n, .is = n, .os = n};
  int howmany = 1;
  fftw::IODim<T> howmany_dim{.n = n, .is = n, .os = n};

  auto pf = fftw::Plan<T>::guru_split_dft_r2c(rank, &dim, howmany, &howmany_dim,
                                              in, ro, io, FFTW_ESTIMATE);
  ASSERT_NE(pf.plan, nullptr);

  auto pb = fftw::Plan<T>::guru_split_dft_c2r(rank, &dim, howmany, &howmany_dim,
                                              ro, io, in, FFTW_ESTIMATE);
  ASSERT_NE(pb.plan, nullptr);
}

TEST(TestHilbert, Correct) {
  using T = double;

  const std::array<T, 10> inp = {
      -0.999984, -0.736924, 0.511211, -0.0826997, 0.0655345,
      -0.562082, -0.905911, 0.357729, 0.358593,   0.869386,
  };
  std::array<T, 10> out{};
  const std::array<T, 10> expect = {
      1.45197493, 1.15365169, 0.54703078, 0.27346519, 0.15097965,
      0.83696245, 1.1476185,  0.71885109, 0.46089151, 1.07384968};

  hilbert_abs<T>(inp, out);
}