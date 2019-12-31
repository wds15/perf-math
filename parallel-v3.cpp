#include <benchmark/benchmark.h>
#include <boost/math/special_functions/gamma.hpp>
#include <cmath>
#include <stan/math/prim/core/init_threadpool_tbb.hpp>
#include <stan/math.hpp>
#include <algorithm>
#include <sstream>
#include <tuple>
#include <vector>

template <typename T>
struct grouped_count_lpdf {
  grouped_count_lpdf() {}

  // does the reduction in the sub-slice start to end
  inline T operator()(std::size_t start, std::size_t end,
                      std::vector<int>& sub_slice, std::vector<T>& lambda,
                      const std::vector<int>& gidx) const {
    const std::size_t num_terms = end - start + 1;
    // std::cout << "sub-slice " << start << " - " << end << "; num_terms = " <<
    // num_terms << "; size = " << sub_slice.size() << std::endl;
    std::vector<T> lambda_slice(num_terms, 0.0);
    for (std::size_t i = 0; i != num_terms; ++i)
      lambda_slice[i] = lambda[gidx[start + i]];
    return stan::math::poisson_lpmf(sub_slice, lambda_slice);
  }
};

using stan::math::var;

void setup_data(std::size_t groups,
                std::size_t elems_per_group,
                std::vector<int>& data, std::vector<int>& gidx) {

  const std::size_t elems = groups * elems_per_group;

  data.clear();
  gidx.clear();

  data.resize(elems);
  gidx.resize(elems);

  for (std::size_t i = 0; i != elems; ++i) {
    data[i] = i;
    gidx[i] = i / elems_per_group;
  }

}

void setup_var(std::size_t groups, std::vector<var>& lambda) {

  lambda.clear();
  lambda.resize(groups);
  
  for (std::size_t i = 0; i != groups; ++i)
    lambda[i] = i + 0.2;
}

const std::size_t elems = 10000;
const std::size_t groups_dense = 10;
const std::size_t groups_sparse = 100; 


// a lot of data per group
static void BM_parallel_dense(benchmark::State& state) {
  stan::math::init_threadpool_tbb();

  const std::size_t groups = groups_dense; 
  const std::size_t elems_per_group = elems / groups;
  
  std::vector<int> data;
  std::vector<int> gidx;

  setup_data(groups, elems_per_group, data, gidx);
  
  for (auto _ : state) {
    std::vector<var> lambda;
    setup_var(groups, lambda);

    var poisson_lpdf = stan::math::parallel_reduce_sum<grouped_count_lpdf<var>>(
        data.begin(), data.end(), var(0.0), 1000, lambda, gidx);

    stan::math::grad(poisson_lpdf.vi_);
    
    stan::math::recover_memory();
  }
  
  stan::math::recover_memory();
}
BENCHMARK(BM_parallel_dense);


static void BM_serial_dense(benchmark::State& state) {

  stan::math::init_threadpool_tbb();

  const std::size_t groups = groups_dense; 
  const std::size_t elems_per_group = elems / groups;
  
  std::vector<int> data;
  std::vector<int> gidx;

  setup_data(groups, elems_per_group, data, gidx);
  
  for (auto _ : state) {
    std::vector<var> lambda;
    setup_var(groups, lambda);

    std::vector<var> lambda_per_elem;
    for (std::size_t i = 0; i != elems; ++i) {
      lambda_per_elem.push_back(lambda[gidx[i]]);
    }

    var poisson_lpdf = stan::math::poisson_lpmf(data, lambda_per_elem);

    stan::math::grad(poisson_lpdf.vi_);
    
    stan::math::recover_memory();
  }
  
  stan::math::recover_memory();
}
BENCHMARK(BM_serial_dense);

// few data per group
static void BM_parallel_sparse(benchmark::State& state) {
  stan::math::init_threadpool_tbb();

  const std::size_t groups = groups_sparse; 
  const std::size_t elems_per_group = elems / groups;
  
  std::vector<int> data;
  std::vector<int> gidx;

  setup_data(groups, elems_per_group, data, gidx);
  
  for (auto _ : state) {
    std::vector<var> lambda;
    setup_var(groups, lambda);

    var poisson_lpdf = stan::math::parallel_reduce_sum<grouped_count_lpdf<var>>(
        data.begin(), data.end(), var(0.0), 1000, lambda, gidx);

    stan::math::grad(poisson_lpdf.vi_);
    
    stan::math::recover_memory();
  }
  
  stan::math::recover_memory();
}
BENCHMARK(BM_parallel_sparse);


static void BM_serial_sparse(benchmark::State& state) {

  stan::math::init_threadpool_tbb();

  const std::size_t groups = groups_sparse; 
  const std::size_t elems_per_group = elems / groups;
  
  std::vector<int> data;
  std::vector<int> gidx;

  setup_data(groups, elems_per_group, data, gidx);
  
  for (auto _ : state) {
    std::vector<var> lambda;
    setup_var(groups, lambda);

    std::vector<var> lambda_per_elem;
    for (std::size_t i = 0; i != elems; ++i) {
      lambda_per_elem.push_back(lambda[gidx[i]]);
    }

    var poisson_lpdf = stan::math::poisson_lpmf(data, lambda_per_elem);

    stan::math::grad(poisson_lpdf.vi_);
    
    stan::math::recover_memory();
  }
  
  stan::math::recover_memory();
}
BENCHMARK(BM_serial_sparse);



BENCHMARK_MAIN();
