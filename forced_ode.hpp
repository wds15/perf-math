
// Code generated by stanc v2.22.1
#include <stan/model/model_header.hpp>
namespace forced_ode_model_namespace {

template <typename T, typename S>
std::vector<T> resize_to_match__(std::vector<T>& dst, const std::vector<S>& src) {
  dst.resize(src.size());
  return dst;
}

template <typename T>
Eigen::Matrix<T, -1, -1>
resize_to_match__(Eigen::Matrix<T, -1, -1>& dst, const Eigen::Matrix<T, -1, -1>& src) {
  dst.resize(src.rows(), src.cols());
  return dst;
}

template <typename T>
Eigen::Matrix<T, 1, -1>
resize_to_match__(Eigen::Matrix<T, 1, -1>& dst, const Eigen::Matrix<T, 1, -1>& src) {
  dst.resize(src.size());
  return dst;
}

template <typename T>
Eigen::Matrix<T, -1, 1>
resize_to_match__(Eigen::Matrix<T, -1, 1>& dst, const Eigen::Matrix<T, -1, 1>& src) {
  dst.resize(src.size());
  return dst;
}
std::vector<double> to_doubles__(std::initializer_list<double> x) {
  return x;
}

std::vector<stan::math::var> to_vars__(std::initializer_list<stan::math::var> x) {
  return x;
}

inline void validate_positive_index(const char* var_name, const char* expr,
                                    int val) {
  if (val < 1) {
    std::stringstream msg;
    msg << "Found dimension size less than one in simplex declaration"
        << "; variable=" << var_name << "; dimension size expression=" << expr
        << "; expression value=" << val;
    std::string msg_str(msg.str());
    throw std::invalid_argument(msg_str.c_str());
  }
}

inline void validate_unit_vector_index(const char* var_name, const char* expr,
                                       int val) {
  if (val <= 1) {
    std::stringstream msg;
    if (val == 1) {
      msg << "Found dimension size one in unit vector declaration."
          << " One-dimensional unit vector is discrete"
          << " but the target distribution must be continuous."
          << " variable=" << var_name << "; dimension size expression=" << expr;
    } else {
      msg << "Found dimension size less than one in unit vector declaration"
          << "; variable=" << var_name << "; dimension size expression=" << expr
          << "; expression value=" << val;
    }
    std::string msg_str(msg.str());
    throw std::invalid_argument(msg_str.c_str());
  }
}


using std::istream;
using std::string;
using std::stringstream;
using std::vector;
using stan::io::dump;
using stan::math::lgamma;
using stan::model::model_base_crtp;
using stan::model::rvalue;
using stan::model::cons_list;
using stan::model::index_uni;
using stan::model::index_max;
using stan::model::index_min;
using stan::model::index_min_max;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::nil_index_list;
using namespace stan::math; 

static int current_statement__ = 0;
static const std::vector<string> locations_array__ = {" (found before start of program)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 3, column 4 to column 22)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 4, column 4 to column 23)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 5, column 4 to column 23)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 6, column 4 to line 7, column 46)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 8, column 4 to column 23)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 12, column 6 to column 51)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 11, column 4 to line 12, column 51)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 14, column 4 to column 44)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 16, column 4 to column 16)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 2, column 77 to line 17, column 3)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 20, column 4 to column 15)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 21, column 4 to column 125)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 19, column 69 to line 22, column 3)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 25, column 4 to column 26)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 26, column 4 to column 81)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 27, column 4 to column 42)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 24, column 80 to line 28, column 3)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 32, column 4 to column 23)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 33, column 4 to column 23)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 34, column 4 to line 35, column 46)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 37, column 4 to column 16)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 31, column 83 to line 38, column 3)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 41, column 4 to column 15)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 42, column 4 to column 111)",
                                                      " (in '../stan_pkpd/amici/forced_ode.stan', line 40, column 91 to line 43, column 3)"};

template <typename T0__, typename T1__, typename T2__, typename T3__>
std::vector<typename boost::math::tools::promote_args<T0__, T1__, T2__,
T3__>::type>
oral_1cmt(const T0__& time, const std::vector<T1__>& y,
          const std::vector<T2__>& theta, const std::vector<T3__>& x_r,
          const std::vector<int>& x_i, std::ostream* pstream__) {
  using local_scalar_t__ = typename boost::math::tools::promote_args<T0__,
          T1__,
          T2__,
          T3__>::type;
  const static bool propto__ = true;
  (void) propto__;
  
  try {
    int D;
    
    current_statement__ = 1;
    D = size(x_r);
    local_scalar_t__ ka;
    
    current_statement__ = 2;
    ka = std::numeric_limits<double>::quiet_NaN();
    current_statement__ = 2;
    ka = theta[(1 - 1)];
    local_scalar_t__ ke;
    
    current_statement__ = 3;
    ke = std::numeric_limits<double>::quiet_NaN();
    current_statement__ = 3;
    ke = theta[(2 - 1)];
    current_statement__ = 4;
    validate_non_negative_index("dydt", "2", 2);
    std::vector<local_scalar_t__> dydt;
    dydt = std::vector<local_scalar_t__>(2, 0);
    
    current_statement__ = 4;
    dydt = stan::math::array_builder<local_scalar_t__>()
        .add((-ka * y[(1 - 1)])).add(((ka * y[(1 - 1)]) - (ke * y[(2 - 1)])))
        .array();
    current_statement__ = 5;
    validate_non_negative_index("dose_terms", "D", D);
    std::vector<local_scalar_t__> dose_terms;
    dose_terms = std::vector<local_scalar_t__>(D, 0);
    
    current_statement__ = 7;
    for (size_t i = 1; i <= D; ++i) {
      current_statement__ = 6;
      assign(dose_terms, cons_list(index_uni(i), nil_index_list()),
        normal_lpdf<false>(time, x_r[(i - 1)], 1),
        "assigning variable dose_terms");}
    current_statement__ = 8;
    assign(dydt, cons_list(index_uni(1), nil_index_list()),
      (dydt[(1 - 1)] + stan::math::exp(log_sum_exp(dose_terms))),
      "assigning variable dydt");
    current_statement__ = 9;
    return dydt;
  } catch (const std::exception& e) {
    stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
  }
  
}

struct oral_1cmt_functor__ {
template <typename T0__, typename T1__, typename T2__, typename T3__>
std::vector<typename boost::math::tools::promote_args<T0__, T1__, T2__,
T3__>::type>
operator()(const T0__& time, const std::vector<T1__>& y,
           const std::vector<T2__>& theta, const std::vector<T3__>& x_r,
           const std::vector<int>& x_i, std::ostream* pstream__)  const 
{
return oral_1cmt(time, y, theta, x_r, x_i, pstream__);
}
};

template <typename T0__, typename T1__>
Eigen::Matrix<typename boost::math::tools::promote_args<T0__,
T1__>::type, -1, -1>
integrate(const std::vector<T0__>& time, const std::vector<T1__>& theta,
          const std::vector<double>& dose_time, std::ostream* pstream__) {
  using local_scalar_t__ = typename boost::math::tools::promote_args<T0__,
          T1__>::type;
  const static bool propto__ = true;
  (void) propto__;
  
  try {
    current_statement__ = 11;
    validate_non_negative_index("x_i", "0", 0);
    std::vector<int> x_i;
    x_i = std::vector<int>(0, 0);
    
    current_statement__ = 12;
    return to_matrix(
             integrate_ode_rk45(oral_1cmt_functor__(),
               stan::math::array_builder<double>().add(0.0).add(0.0).array(),
               (time[(1 - 1)] - 1E-6), time, theta, dose_time, x_i,
               pstream__, 1E-6, 1E-6, 1E3));
  } catch (const std::exception& e) {
    stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
  }
  
}

struct integrate_functor__ {
template <typename T0__, typename T1__>
Eigen::Matrix<typename boost::math::tools::promote_args<T0__,
T1__>::type, -1, -1>
operator()(const std::vector<T0__>& time, const std::vector<T1__>& theta,
           const std::vector<double>& dose_time, std::ostream* pstream__)  const 
{
return integrate(time, theta, dose_time, pstream__);
}
};

template <bool propto__, typename T0__, typename T1__, typename T2__>
typename boost::math::tools::promote_args<T0__, T1__,
T2__>::type
subject_lpdf(const std::vector<T0__>& y, const std::vector<T1__>& time,
             const std::vector<T2__>& theta,
             const std::vector<double>& dose_time, std::ostream* pstream__) {
  using local_scalar_t__ = typename boost::math::tools::promote_args<T0__,
          T1__,
          T2__>::type;
  
  try {
    int num_obs;
    
    current_statement__ = 14;
    num_obs = size(y);
    current_statement__ = 15;
    validate_non_negative_index("main_cmt", "num_obs", num_obs);
    std::vector<local_scalar_t__> main_cmt;
    main_cmt = std::vector<local_scalar_t__>(num_obs, 0);
    
    current_statement__ = 15;
    assign(main_cmt, nil_index_list(),
      to_array_1d(
        rvalue(integrate(time, theta, dose_time, pstream__),
          cons_list(index_omni(), cons_list(index_uni(2), nil_index_list())),
          "integrate(time, theta, dose_time)")),
      "assigning variable main_cmt");
    current_statement__ = 16;
    return normal_lpdf<false>(y, main_cmt, 0.5);
  } catch (const std::exception& e) {
    stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
  }
  
}

struct subject_lpdf_functor__ {
template <bool propto__, typename T0__, typename T1__, typename T2__>
typename boost::math::tools::promote_args<T0__, T1__,
T2__>::type
operator()(const std::vector<T0__>& y, const std::vector<T1__>& time,
           const std::vector<T2__>& theta,
           const std::vector<double>& dose_time, std::ostream* pstream__)  const 
{
return subject_lpdf(y, time, theta, dose_time, pstream__);
}
};

template <typename T0__, typename T1__, typename T2__, typename T3__>
std::vector<typename boost::math::tools::promote_args<T0__, T1__, T2__,
T3__>::type>
oral_1cmt_basic(const T0__& time, const std::vector<T1__>& y,
                const std::vector<T2__>& theta, const std::vector<T3__>& x_r,
                const std::vector<int>& x_i, std::ostream* pstream__) {
  using local_scalar_t__ = typename boost::math::tools::promote_args<T0__,
          T1__,
          T2__,
          T3__>::type;
  const static bool propto__ = true;
  (void) propto__;
  
  try {
    local_scalar_t__ ka;
    
    current_statement__ = 18;
    ka = std::numeric_limits<double>::quiet_NaN();
    current_statement__ = 18;
    ka = theta[(1 - 1)];
    local_scalar_t__ ke;
    
    current_statement__ = 19;
    ke = std::numeric_limits<double>::quiet_NaN();
    current_statement__ = 19;
    ke = theta[(2 - 1)];
    current_statement__ = 20;
    validate_non_negative_index("dydt", "2", 2);
    std::vector<local_scalar_t__> dydt;
    dydt = std::vector<local_scalar_t__>(2, 0);
    
    current_statement__ = 20;
    dydt = stan::math::array_builder<local_scalar_t__>()
        .add((-ka * y[(1 - 1)])).add(((ka * y[(1 - 1)]) - (ke * y[(2 - 1)])))
        .array();
    current_statement__ = 21;
    return dydt;
  } catch (const std::exception& e) {
    stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
  }
  
}

struct oral_1cmt_basic_functor__ {
template <typename T0__, typename T1__, typename T2__, typename T3__>
std::vector<typename boost::math::tools::promote_args<T0__, T1__, T2__,
T3__>::type>
operator()(const T0__& time, const std::vector<T1__>& y,
           const std::vector<T2__>& theta, const std::vector<T3__>& x_r,
           const std::vector<int>& x_i, std::ostream* pstream__)  const 
{
return oral_1cmt_basic(time, y, theta, x_r, x_i, pstream__);
}
};

template <typename T0__, typename T1__, typename T2__>
std::vector<std::vector<typename boost::math::tools::promote_args<T0__, T1__,
T2__>::type>>
integrate_bdf_basic(const std::vector<T0__>& time,
                    const std::vector<T1__>& y0,
                    const std::vector<T2__>& theta,
                    const std::vector<double>& dose_time,
                    std::ostream* pstream__) {
  using local_scalar_t__ = typename boost::math::tools::promote_args<T0__,
          T1__,
          T2__>::type;
  const static bool propto__ = true;
  (void) propto__;
  
  try {
    current_statement__ = 23;
    validate_non_negative_index("x_i", "0", 0);
    std::vector<int> x_i;
    x_i = std::vector<int>(0, 0);
    
    current_statement__ = 24;
    return integrate_ode_bdf(oral_1cmt_basic_functor__(), y0,
             (time[(1 - 1)] - 1E-6), time, theta, dose_time, x_i, pstream__,
             1E-6, 1E-6, 1E3);
  } catch (const std::exception& e) {
    stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
  }
  
}

struct integrate_bdf_basic_functor__ {
template <typename T0__, typename T1__, typename T2__>
std::vector<std::vector<typename boost::math::tools::promote_args<T0__, T1__,
T2__>::type>>
operator()(const std::vector<T0__>& time, const std::vector<T1__>& y0,
           const std::vector<T2__>& theta,
           const std::vector<double>& dose_time, std::ostream* pstream__)  const 
{
return integrate_bdf_basic(time, y0, theta, dose_time, pstream__);
}
};

class forced_ode_model : public model_base_crtp<forced_ode_model> {

 private:
  int pos__;
 
 public:
  ~forced_ode_model() { }
  
  std::string model_name() const { return "forced_ode_model"; }
  
  forced_ode_model(stan::io::var_context& context__,
                   unsigned int random_seed__ = 0,
                   std::ostream* pstream__ = nullptr) : model_base_crtp(0) {
    typedef double local_scalar_t__;
    boost::ecuyer1988 base_rng__ = 
        stan::services::util::create_rng(random_seed__, 0);
    (void) base_rng__;  // suppress unused var warning
    static const char* function__ = "forced_ode_model_namespace::forced_ode_model";
    (void) function__;  // suppress unused var warning
    
    try {
      
      pos__ = 1;
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    num_params_r__ = 0U;
    
    try {
      
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
  }
  template <bool propto__, bool jacobian__, typename T__>
  T__ log_prob(std::vector<T__>& params_r__, std::vector<int>& params_i__,
               std::ostream* pstream__ = 0) const {
    typedef T__ local_scalar_t__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    static const char* function__ = "forced_ode_model_namespace::log_prob";
(void) function__;  // suppress unused var warning

    stan::io::reader<local_scalar_t__> in__(params_r__, params_i__);
    
    try {
      
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
    } // log_prob() 
    
  template <typename RNG>
  void write_array(RNG& base_rng__, std::vector<double>& params_r__,
                   std::vector<int>& params_i__, std::vector<double>& vars__,
                   bool emit_transformed_parameters__ = true,
                   bool emit_generated_quantities__ = true,
                   std::ostream* pstream__ = 0) const {
    typedef double local_scalar_t__;
    vars__.resize(0);
    stan::io::reader<local_scalar_t__> in__(params_r__, params_i__);
    static const char* function__ = "forced_ode_model_namespace::write_array";
(void) function__;  // suppress unused var warning

    (void) function__;  // suppress unused var warning

    double lp__ = 0.0;
    (void) lp__;  // dummy to suppress unused var warning
    stan::math::accumulator<double> lp_accum__;
    
    try {
      if (logical_negation((primitive_value(emit_transformed_parameters__) ||
            primitive_value(emit_generated_quantities__)))) {
        return ;
      } 
      if (logical_negation(emit_generated_quantities__)) {
        return ;
      } 
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    } // write_array() 
    
  void transform_inits(const stan::io::var_context& context__,
                       std::vector<int>& params_i__,
                       std::vector<double>& vars__, std::ostream* pstream__) const {
    typedef double local_scalar_t__;
    vars__.resize(0);
    vars__.reserve(num_params_r__);
    
    try {
      int pos__;
      
      pos__ = 1;
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    } // transform_inits() 
    
  void get_param_names(std::vector<std::string>& names__) const {
    
    names__.resize(0);
    
    } // get_param_names() 
    
  void get_dims(std::vector<std::vector<size_t>>& dimss__) const {
    dimss__.resize(0);
    std::vector<size_t> dims__;
    dimss__.push_back(dims__);
    dims__.resize(0);
    
    } // get_dims() 
    
  void constrained_param_names(std::vector<std::string>& param_names__,
                               bool emit_transformed_parameters__ = true,
                               bool emit_generated_quantities__ = true) const {
    
    
    if (emit_transformed_parameters__) {
      
    }
    
    if (emit_generated_quantities__) {
      
    }
    
    } // constrained_param_names() 
    
  void unconstrained_param_names(std::vector<std::string>& param_names__,
                                 bool emit_transformed_parameters__ = true,
                                 bool emit_generated_quantities__ = true) const {
    
    
    if (emit_transformed_parameters__) {
      
    }
    
    if (emit_generated_quantities__) {
      
    }
    
    } // unconstrained_param_names() 
    
  std::string get_constrained_sizedtypes() const {
    stringstream s__;
    s__ << "[]";
    return s__.str();
    } // get_constrained_sizedtypes() 
    
  std::string get_unconstrained_sizedtypes() const {
    stringstream s__;
    s__ << "[]";
    return s__.str();
    } // get_unconstrained_sizedtypes() 
    
  
    // Begin method overload boilerplate
    template <typename RNG>
    void write_array(RNG& base_rng__,
                     Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                     Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                     bool emit_transformed_parameters__ = true,
                     bool emit_generated_quantities__ = true,
                     std::ostream* pstream = 0) const {
      std::vector<double> params_r_vec(params_r.size());
      for (int i = 0; i < params_r.size(); ++i)
        params_r_vec[i] = params_r(i);
      std::vector<double> vars_vec;
      std::vector<int> params_i_vec;
      write_array(base_rng__, params_r_vec, params_i_vec, vars_vec,
          emit_transformed_parameters__, emit_generated_quantities__, pstream);
      vars.resize(vars_vec.size());
      for (int i = 0; i < vars.size(); ++i)
        vars(i) = vars_vec[i];
    }

    template <bool propto__, bool jacobian__, typename T_>
    T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
               std::ostream* pstream = 0) const {
      std::vector<T_> vec_params_r;
      vec_params_r.reserve(params_r.size());
      for (int i = 0; i < params_r.size(); ++i)
        vec_params_r.push_back(params_r(i));
      std::vector<int> vec_params_i;
      return log_prob<propto__,jacobian__,T_>(vec_params_r, vec_params_i, pstream);
    }

    void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
                         std::ostream* pstream__) const {
      std::vector<double> params_r_vec;
      std::vector<int> params_i_vec;
      transform_inits(context, params_i_vec, params_r_vec, pstream__);
      params_r.resize(params_r_vec.size());
      for (int i = 0; i < params_r.size(); ++i)
        params_r(i) = params_r_vec[i];
    }

};
}
typedef forced_ode_model_namespace::forced_ode_model stan_model;

#ifndef USING_R

// Boilerplate
stan::model::model_base& new_model(
        stan::io::var_context& data_context,
        unsigned int seed,
        std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}

#endif

