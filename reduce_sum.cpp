#include <benchmark/benchmark.h>

#include <stan/math.hpp>

#include <algorithm>
#include <sstream>
#include <tuple>
#include <vector>


// reduce functor which is the BinaryFunction
// here we use iterators which represent integer indices
template <typename T>
struct count_lpdf {
  count_lpdf() {}

  // does the reduction in the sub-slice start to end
  inline T operator()(std::size_t start, std::size_t end,
                      const std::vector<int>& sub_slice,
                      const std::vector<T>& lambda,
                      const std::vector<int>& idata) const {
    return stan::math::poisson_lpmf(sub_slice, lambda[0]);
  }
};


static void BM_reduce_sum_dd(benchmark::State& state) {
  stan::math::init_threadpool_tbb();

  double lambda_d = 10.0;
  const std::size_t elems = 10000;
  std::vector<int> data(elems);

  for (std::size_t i = 0; i != elems; ++i)
    data[i] = i;

  std::vector<int> idata;
  
  for (auto _ : state) {    
    std::vector<double> vlambda_d(1, lambda_d);

    double poisson_lpdf = stan::math::reduce_sum<count_lpdf<double>>(
        data, 0.0, 5, vlambda_d, idata);
  }
  stan::math::recover_memory();
}
//BENCHMARK(BM_reduce_sum_dd);


static void BM_reduce_sum_dv(benchmark::State& state) {
  stan::math::init_threadpool_tbb();
  
  double lambda_d = 10.0;
  const std::size_t elems = 10000;
  const std::size_t num_iter = 1000;
  std::vector<int> data(elems);

  for (std::size_t i = 0; i != elems; ++i)
    data[i] = i;

  using stan::math::var;

  std::vector<int> idata;

  for (auto _ : state) {    

    std::vector<var> vlambda_v(1, lambda_d);

    var poisson_lpdf = stan::math::reduce_sum<count_lpdf<var>>(
        data, var(0.0), 5, vlambda_v, idata);
    
    stan::math::grad(poisson_lpdf.vi_);

    stan::math::recover_memory();
  }
  
  stan::math::recover_memory();
}
//BENCHMARK(BM_reduce_sum_dv);


template <typename T>
struct grouped_count_lpdf {
  grouped_count_lpdf() {}

  // does the reduction in the sub-slice start to end
  inline T operator()(std::size_t start, std::size_t end,
                      const std::vector<int>& sub_slice,
                      const std::vector<T>& lambda,
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


static void BM_reduce_sum_hierarchial_dv(benchmark::State& state) {
  stan::math::init_threadpool_tbb();
  
  double lambda_d = 10.0;
  const std::size_t groups = 10;
  const std::size_t elems_per_group = 1000;
  const std::size_t elems = groups * elems_per_group;
  const std::size_t num_iter = 1000;

  std::vector<int> data(elems);
  std::vector<int> gidx(elems);

  for (std::size_t i = 0; i != elems; ++i) {
    data[i] = i;
    gidx[i] = i / elems_per_group;
  }

  using stan::math::var;

  for (auto _ : state) {    

    std::vector<var> vlambda_v;

    for (std::size_t i = 0; i != groups; ++i)
      vlambda_v.push_back(i + 0.2);

    var lambda_v = vlambda_v[0];

    var poisson_lpdf = stan::math::reduce_sum<grouped_count_lpdf<var>>(
        data, var(0.0), 5, vlambda_v, gidx);

    stan::math::grad(poisson_lpdf.vi_);
    
    stan::math::recover_memory();
  }
  
  stan::math::recover_memory();
}
//BENCHMARK(BM_reduce_sum_hierarchial_dv);


template <typename T>
struct slice_group_count_lpdf {
  slice_group_count_lpdf() {}

  // does the reduction in the sub-slice start to end
  inline T operator()(std::size_t start, std::size_t end,
                      const std::vector<T>& lambda_slice,
                      const std::vector<int>& y,
                      const std::vector<int>& gsidx) const {
    const std::size_t num_groups = end - start + 1;
    T result = 0.0;
    for (std::size_t i = 0; i != num_groups; ++i) {
      std::vector<int> y_group(y.begin() + gsidx[start + i],
                               y.begin() + gsidx[start + i + 1]);
      result += stan::math::poisson_lpmf(y_group, lambda_slice[i]);
    }
    return result;
  }
};

static void BM_reduce_sum_hierarchial_vd(benchmark::State& state) {
  stan::math::init_threadpool_tbb();

  double lambda_d = 10.0;
  const std::size_t groups = 100;
  const std::size_t elems_per_group = 100;
  const std::size_t elems = groups * elems_per_group;

  std::vector<int> data(elems);
  std::vector<int> gidx(elems);
  std::vector<int> gsidx(groups + 1);

  for (std::size_t i = 0, k = 0; i != groups; ++i) {
    gsidx[i] = k;
    for (std::size_t j = 0; j != elems_per_group; ++j, ++k) {
      data[k] = k;
      gidx[k] = i;
    }
    gsidx[i + 1] = k;
  }

  using stan::math::var;

  for (auto _ : state) {
    std::vector<var> vlambda_v;

    for (std::size_t i = 0; i != groups; ++i)
      vlambda_v.push_back(i + 0.2);

    var poisson_lpdf = stan::math::reduce_sum<slice_group_count_lpdf<var>>(
        vlambda_v, var(0.0), 5, data, gsidx);
    
    stan::math::grad(poisson_lpdf.vi_);
    stan::math::recover_memory();
  }
  
  stan::math::recover_memory();
}
BENCHMARK(BM_reduce_sum_hierarchial_vd);


BENCHMARK_MAIN();
