#ifndef EKF_SAMPLER_HPP_
#define EKF_SAMPLER_HPP_

#include "epf/core/enum.hpp"
#include "epf/core/measurement.hpp"
#include "epf/core/process.hpp"
#include "epf/core/state.hpp"
#include "epf/util/math.hpp"
#include <Eigen/Dense>
#include <random>
#include <range/v3/view/zip.hpp>
#include <type_traits>
#include <vector>

namespace epf {

template <typename State, typename Output>
struct CenterDiff {
  using StateTrait  = StateTraits<State>;
  using OutputTrait = StateTraits<Output>;
  using OutputNoise = NoiseTraits<Output>;

  using StateVector       = typename StateTrait::ArithmeticType;
  using OutputNoiseVector = typename OutputNoise::ArithmeticType;

  template <std::size_t Row, std::size_t Col>
  using Matrix = Eigen::Matrix<typename StateTrait::ValueType, Row, Col>;

  static inline constexpr auto STATE_DIM  = StateTrait::Dimension::value;
  static inline constexpr auto OUTPUT_DIM = OutputTrait::Dimension::value;
  static inline constexpr auto EPSILON    = 1e-5;

  using StateJacobian       = Matrix<STATE_DIM, STATE_DIM>;
  using MeasurementJacobian = Matrix<OUTPUT_DIM, OUTPUT_DIM>;

  [[nodiscard]] StateJacobian calculate_state_jacobian(ProcessModel<State>* const t_proc, StateVector const& t_vec) {
    StateJacobian ret_val;

    for (std::size_t i = 0; i < t_vec.size(); ++i) {
      StateVector epsilon = StateVector::Zero();
      epsilon[i]          = EPSILON;

      std::vector diff_pt{t_vec + epsilon, t_vec - epsilon};
      t_proc->predict(diff_pt);
      ret_val.col(i) = (diff_pt[0] - diff_pt[1]) / (2 * EPSILON);
    }

    return ret_val;
  }

  [[nodiscard]] MeasurementJacobian calculate_output_jacobian(MeasurementModel<State>* const t_meas,
                                                              StateVector const& t_vec) {
    MeasurementJacobian ret_val;

    for (std::size_t i = 0; i < t_vec.size(); ++i) {
      StateVector epsilon = StateVector::Zero();
      epsilon[i]          = EPSILON;

      std::vector<StateVector> diff_pt{t_vec + epsilon, t_vec - epsilon};
      std::vector<OutputNoiseVector> noise(2, OutputNoiseVector::Zero());
      t_meas->predict(diff_pt, noise);
      ret_val.col(i) = (diff_pt[0] - diff_pt[1]) / (2 * EPSILON);
    }

    return ret_val;
  }
};

template <typename State, typename Output, typename JacobianStrategy = CenterDiff<State, Output>>
class EKFSampler : private JacobianStrategy {
  using StateTrait  = StateTraits<State>;
  using StateNoise  = NoiseTraits<State>;
  using OutputTrait = StateTraits<Output>;
  using OutputNoise = NoiseTraits<Output>;

  using StateNoiseVector = typename StateNoise::ArithmeticType;
  using StateVector      = typename StateTrait::ArithmeticType;
  using StateCovariance  = typename StateTrait::CovarianceType;

  template <std::size_t Row, std::size_t Col>
  using Matrix = Eigen::Matrix<typename StateTrait::ValueType, Row, Col>;

  static inline constexpr auto STATE_DIM  = StateTrait::Dimension::value;
  static inline constexpr auto OUTPUT_DIM = OutputTrait::Dimension::value;

  using StateJacobian       = Matrix<STATE_DIM, STATE_DIM>;
  using MeasurementJacobian = Matrix<OUTPUT_DIM, OUTPUT_DIM>;

  using JacobianStrategy::calculate_output_jacobian;
  using JacobianStrategy::calculate_state_jacobian;

  std::vector<StateCovariance> covariances_;

 public:
  SamplingResult importance_sampling(ProcessModel<State>* const t_proc, MeasurementModel<State>* const t_meas,
                                     std::vector<StateVector>& t_prev, std::vector<double>& t_weight) {
    auto const& Q = t_proc->get_noise_covariance();
    auto const& R = t_meas->get_noise_covariance();

    auto const linearization_pt = t_prev;

    t_proc->predict(t_prev);  // time update
    auto const output_predict = t_meas->predict(t_prev);
    for (auto [pt, cov, state, out] :
         ranges::views::zip(linearization_pt, this->covariances_, t_prev, output_predict)) {
      StateJacobian const F       = this->calculate_state_jacobian(t_proc, pt);
      MeasurementJacobian const H = this->calculate_output_jacobian(t_meas, pt);

      // @note: in Anderson and Moore 1979 (optimal filtering), G = df(v)/dv (0) needs to be calculated
      cov = F * cov * F.transpose() + Q;

      Matrix<STATE_DIM, OUTPUT_DIM> K = cov * H.transpose() * (R + H * cov * H.transpose()).inverse();
      state += K * (t_meas->get_latest_output() - out);
      cov = (StateCovariance::Identity() - K * H) * cov;
    }
  }
};

}  // namespace epf

#endif