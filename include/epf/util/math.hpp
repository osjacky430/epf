#ifndef EPF_MATH_HPP_
#define EPF_MATH_HPP_

#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <random>

namespace epf {

inline constexpr auto constraint_angle(double const t_in) {
  if (t_in < -boost::math::double_constants::pi) {
    return t_in + boost::math::double_constants::two_pi;
  }

  if (t_in > boost::math::double_constants::pi) {
    return t_in - boost::math::double_constants::two_pi;
  }

  return t_in;
}

template <std::size_t Dim, typename T = double>
class MultivariateNormalDistribution {
  static inline constexpr int Size = Dim;
  Eigen::Matrix<T, Size, 1> mean_;
  Eigen::Matrix<T, Size, Size> transform_;

  std::normal_distribution<> dist_;

  static Eigen::Matrix<T, Size, Size> covaraince_transform(Eigen::Matrix<T, Size, Size> const& t_cov) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Size, Size>> eigen_solver(t_cov);
    return eigen_solver.eigenvectors() * eigen_solver.eigenvalues().cwiseSqrt().asDiagonal();
  }

 public:
  explicit MultivariateNormalDistribution(Eigen::Matrix<T, Size, 1> const& t_mean,
                                          Eigen::Matrix<T, Size, Size> const& t_cov)
    : mean_{t_mean}, transform_(covaraince_transform(t_cov)) {}

  template <typename Generator>
  Eigen::Matrix<T, Size, 1> operator()(Generator& t_generator) noexcept {
    auto const single_rnv = [&](auto /**/) { return this->dist_(t_generator); };
    return this->mean_ + this->transform_ * (Eigen::Matrix<T, Size, 1>{}.unaryExpr(single_rnv));
  }
};

inline std::vector<double> calculate_cumulative_weight(std::vector<double> const& t_weight) {
  std::vector<double> ret_val = t_weight;
  for (std::size_t i = 1; i < t_weight.size(); ++i) {
    ret_val[i] += ret_val[i - 1];
  }

  return ret_val;
}

inline constexpr auto weighted_sum = [](auto const& t_state, auto const& t_weight_state_pair) noexcept {
  auto const& [state, weight] = t_weight_state_pair;
  return t_state + weight * state;
};

}  // namespace epf

#endif