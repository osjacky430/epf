#ifndef STATE_HPP_
#define STATE_HPP_

#include <Eigen/Dense>
#include <boost/concept/assert.hpp>
#include <boost/concept/usage.hpp>
#include <type_traits>

namespace epf {

template <typename State>
struct NoiseTraits {
  using ValueType = typename State::ValueType;
  using Dimension = typename State::NoiseDimension;

  using ArithmeticType = Eigen::Vector<ValueType, Dimension::value>;
  using NoiseCovType   = Eigen::Matrix<ValueType, Dimension::value, Dimension::value>;
};

template <typename, typename T = std::void_t<>>
struct satisfy_noise_traits {
  static inline constexpr bool value = false;
};

template <typename State>
struct satisfy_noise_traits<State, std::void_t<typename State::NoiseDimension, typename State::ValueType>> {
  static inline constexpr bool value = true;
};

template <typename, typename T = std::void_t<>>
struct satisfy_state_traits {
  static inline constexpr bool value = false;
};

template <typename State>
struct satisfy_state_traits<State, std::void_t<typename State::Dimension, typename State::ValueType>> {
  static inline constexpr bool value = true;
};

template <typename State>
struct StateTraits {
  using ValueType = typename State::ValueType;
  using Dimension = typename State::Dimension;

  using ArithmeticType = Eigen::Vector<ValueType, Dimension::value>;
  using CovarianceType = Eigen::Matrix<ValueType, Dimension::value, Dimension::value>;
};

template <typename State, typename Tag = typename State::TagType>
State from_state_vector(typename StateTraits<State>::ArithmeticType const&);

template <typename State, typename Tag = typename State::TagType>
typename StateTraits<State>::ArithmeticType to_state_vector(State const&);

template <typename State, typename Tag = typename State::TagType>
typename State::ValueType& x_coor(typename StateTraits<State>::ArithmeticType&);

template <typename State, typename Tag = typename State::TagType>
typename State::ValueType& y_coor(typename StateTraits<State>::ArithmeticType&);

template <typename State, typename Tag = typename State::TagType>
typename State::ValueType& w_coor(typename StateTraits<State>::ArithmeticType&);

template <typename State>
struct BasicStateConcept {  // NOLINT(*-special-member-functions), because this class is not meant to be instantiated
  BOOST_CONCEPT_USAGE(BasicStateConcept) {
    using StateVector = typename StateTraits<State>::ArithmeticType;
    State s1          = from_state_vector<State>(StateVector{});
    StateVector sv    = to_state_vector(s1);
  }
};

}  // namespace epf

#endif