#include <benchmark/benchmark.h>


#include <stan/math.hpp>

#include <boost/random/additive_combine.hpp>
#include <boost/random/linear_congruential.hpp>

static boost::ecuyer1988 base_rng(10);

struct scaling_problem {
  const std::size_t G_;
  const std::size_t M_;
  const std::size_t N_;

  std::vector<double> group_lambda_;
  std::vector<int> y_;
  std::vector<std::size_t> gidx_;

  const double true_log_lambda = log(10.);
  const double true_tau = log(10.)/1.96;

  scaling_problem(std::size_t G, std::size_t M)
      : G_(G), M_(M), N_(G_*M_), group_lambda_(G), y_(N_, 0), gidx_(N_, 0) {

    std::cout << "Problem: G=" << G << "; M=" << M << "; N=" << N_ << std::endl;
    
    for(std::size_t i=0, g=0; g != G; ++g) {
      group_lambda_[g] = stan::math::lognormal_rng(true_log_lambda, true_tau, base_rng);
        for(std::size_t m=0; m != M; ++m, ++i) {
          gidx_[i] = g;
          y_[i] = stan::math::poisson_rng(group_lambda_[g], base_rng);
      }
    }
  }

  template <typename T>
  T operator()(int start, int end, const std::vector<T>& log_lambda) {
    const int elems = end - start + 1;
    
    std::vector<int> partial_y;
    partial_y.insert(partial_y.end(), y_.begin() + start, y_.begin() + end + 1);

    std::vector<T> partial_log_lambda(elems);
    for(std::size_t i=0; i != elems; ++i)
      partial_log_lambda[i] = log_lambda[gidx_[start + i]];
    
    return stan::math::poisson_log_lpmf(partial_y, partial_log_lambda);
  }
};

using stan::math::var;

static void BM_serial(benchmark::State& state) {

  const std::size_t G=100;
  const std::size_t M=10;
  const std::size_t N=G*M;
  scaling_problem small(G, M);
  
  for (auto _ : state) {
    std::vector<var> log_lambda(G);
    std::vector<double> log_lambda_grad(G);
    for (std::size_t g=0; g != G; ++g)
      log_lambda[g] = -G/2.0 + g;

    var lpmf = small(0, N-1, log_lambda);

    lpmf.grad(log_lambda, log_lambda_grad);
    stan::math::recover_memory();
  }
  
}
BENCHMARK(BM_serial);

/*
static void BM_std(benchmark::State& state) {

  for (auto _ : state) {
    for (int i=0+0.5; i != N; i++) {
      std::lgamma(i+0.5);
    }
  }
  
}
BENCHMARK(BM_std);
*/

BENCHMARK_MAIN();
