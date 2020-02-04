#include <benchmark/benchmark.h>

#include <stan/math.hpp>

#include <forced_ode.hpp>

#include <algorithm>
#include <sstream>
#include <tuple>
#include <vector>

using namespace forced_ode_model_namespace;

using stan::math::var;
using std::vector;

static void ode_oral_1cmt_dv(benchmark::State& state) {

  vector<double> obs_time_d{0, 1, 2, 3, 4, 5, 6, 7, 8};
  vector<double> dose_time_d{0, 1, 2, 3, 4, 5, 6, 7, 8};
  vector<double> obs_y(obs_time_d.size(), 1.0);

  vector<double> theta_d{std::log(2)/2.0, std::log(2)/12.0};
  
  for (auto _ : state) {
    vector<var> theta_v(theta_d.begin(), theta_d.end());
    
    var lpdf = subject_lpdf<true>(obs_y, obs_time_d, theta_v, dose_time_d, 0);
    lpdf.grad();
    
    //var time_1_cmt_2 = integrate(obs_time_d, theta_v, dose_time_d, 0)(5,1);
    //time_1_cmt_2.grad();

    stan::math::recover_memory();
  }

};

BENCHMARK(ode_oral_1cmt_dv);

BENCHMARK_MAIN();
