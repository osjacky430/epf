#ifndef TIME_SERIES_HPP_
#define TIME_SERIES_HPP_

#include "core/state.hpp"
#include <boost/math/constants/constants.hpp>
#include <random>
#include <vector>

#include <iostream>

namespace time_series {

struct F /*name of the struct TBD*/ {
  struct State {
    struct Tag {};

    using TagType        = Tag;
    using ValueType      = double;
    using NoiseDimension = std::integral_constant<std::size_t, 1>;
    using Dimension      = std::integral_constant<std::size_t, 1>;

    double x_;
  };

  struct Output {
    struct Tag {};

    using TagType        = Tag;
    using ValueType      = double;
    using Dimension      = std::integral_constant<std::size_t, 1>;
    using NoiseDimension = std::integral_constant<std::size_t, 1>;

    double x_;
  };

  static inline constexpr auto TIME_STEP = 60;
  static inline constexpr auto PI        = boost::math::double_constants::pi;
  static inline constexpr double OMEGA   = 4e-2;
  static inline constexpr double PHI_1   = 0.5;
  static inline constexpr double PHI_2   = 0.2;
  static inline constexpr double PHI_3   = 0.5;

  // c++ gamma distribution takes parameter alpha, beta but in fact, it means k and theta..
  // From bayesian statistics point of view, k = alpha, and theta = 1.0 / beta..
  // in paper, state transition is subjected to Ga(3, 2), i.e. alpha = 3, beta = 2.
  // From c++ gamma distribution perspective, alpha = k = 3, beta = theta = 1 / 2
  static inline constexpr double ALPHA = 3.0;
  static inline constexpr double BETA  = 0.5;

  /*
   *  @brief  First and second moment of noise random variable
   */
  static inline constexpr auto STATE_NOISE_MEAN = ALPHA * BETA;
  static inline constexpr auto STATE_NOISE_COV  = ALPHA * BETA * BETA;

  static inline constexpr double OBSERVATION_NOISE = 1e-5;

  std::vector<State> time_series_;
  std::vector<Output> observed_;

  std::mt19937 rng_{std::random_device{}()};
  std::gamma_distribution<> gamm_dist_{ALPHA, BETA};
  std::normal_distribution<> norm_dist_{0.0, std::sqrt(OBSERVATION_NOISE)};

  static Output observation(State const& t_curr, std::size_t const t_time, double const t_noise = 0.0) noexcept {
    if (t_time > TIME_STEP / 2) {
      return Output{t_curr.x_ * PHI_3 - 2 + t_noise};
    }

    return Output{t_curr.x_ * t_curr.x_ * PHI_2 + t_noise};
  }

  static State state_transition(State const& t_prev, std::size_t const t_time, double const t_noise = 0.0) noexcept {
    return State{PHI_1 * t_prev.x_ + 1 + std::sin(OMEGA * PI * static_cast<double>(t_time)) + t_noise};
  };

  // e.g. t_end = 60, t_start = 1 -> time_series_[n] = state at t = n
  F(State const& t_initial_cond, std::size_t const t_end, std::size_t t_start = 0)
    : time_series_(t_start + t_end, t_initial_cond), observed_(t_start + t_end) {
    for (std::size_t i = t_start + 1; i < t_start + t_end; ++i) {
      this->time_series_.at(i) = state_transition(this->time_series_.at(i - 1), i, this->gamm_dist_(this->rng_));
      this->observed_.at(i)    = observation(this->time_series_.at(i), i, this->norm_dist_(this->rng_));
    }
  }

  [[nodiscard]] std::vector<State> const& get_time_series() const noexcept { return this->time_series_; }

  [[nodiscard]] Output get_measurement(std::size_t const t_time) noexcept { return this->observed_.at(t_time); }
};

}  // namespace time_series

namespace epf {

template <>
inline time_series::F::State from_state_vector<time_series::F::State>(
  typename StateTraits<time_series::F::State>::ArithmeticType const& t_v) {
  return time_series::F::State{t_v[0]};
}

template <>
inline typename StateTraits<time_series::F::State>::ArithmeticType to_state_vector(
  time_series::F::State const& t_state) {
  return typename StateTraits<time_series::F::State>::ArithmeticType{t_state.x_};
}

template <>
inline time_series::F::Output from_state_vector<time_series::F::Output>(
  typename StateTraits<time_series::F::Output>::ArithmeticType const& t_val) {
  return time_series::F::Output{t_val[0]};
}

template <>
inline typename StateTraits<time_series::F::Output>::ArithmeticType to_state_vector(
  const time_series::F::Output& t_output) {
  return typename StateTraits<time_series::F::Output>::ArithmeticType{t_output.x_};
}

}  // namespace epf

#endif