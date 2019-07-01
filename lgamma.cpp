#define _REENTRANT

#include <benchmark/benchmark.h>
#include <boost/math/special_functions/gamma.hpp>
#include <cmath>


const int N = 10000;

using namespace boost::math::policies;
typedef policy<promote_double<false> > fast_policy_t;

typedef policy<promote_double<false>, digits10<9> > very_fast_policy_t;


typedef boost::math::policies::policy<
    boost::math::policies::overflow_error<
        boost::math::policies::errno_on_error>,
    boost::math::policies::pole_error<boost::math::policies::errno_on_error>,
  boost::math::policies::promote_double<false> > boost_policy_t;


static void BM_boost(benchmark::State& state) {

  for (auto _ : state) {
    for (int i=0; i != N; i++) {
      boost::math::lgamma(i +0.5, boost_policy_t());
    }
  }
}

//BENCHMARK(BM_boost);

static void BM_boost9(benchmark::State& state) {

  for (auto _ : state) {
    for (int i=0; i != N; i++) {
      boost::math::lgamma(i +0.5, very_fast_policy_t());
    }
  }
  
}
//BENCHMARK(BM_boost9);

static void BM_std(benchmark::State& state) {

  for (auto _ : state) {
    for (int i=0+0.5; i != N; i++) {
      std::lgamma(i+0.5);
    }
  }
  
}
BENCHMARK(BM_std);

static void BM_cstd(benchmark::State& state) {

  int sign = 0;
  
  for (auto _ : state) {
    for (int i=0+0.5; i != N; i++) {
      lgamma_r(i+0.5, &sign);
    }
  }
  
}
BENCHMARK(BM_cstd);

BENCHMARK_MAIN();
