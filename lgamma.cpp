#include <benchmark/benchmark.h>
#include <boost/math/special_functions/gamma.hpp>
#include <cmath>

const int N = 10000;

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
BENCHMARK(BM_boost);

static void BM_std(benchmark::State& state) {

  for (auto _ : state) {
    for (int i=0+0.5; i != N; i++) {
      std::lgamma(i+0.5);
    }
  }
  
}
BENCHMARK(BM_std);

BENCHMARK_MAIN();
