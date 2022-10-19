#ifndef UKF_SAMPLER_HPP_
#define UKF_SAMPLER_HPP_

#include "epf/core/enum.hpp"
#include "epf/core/measurement.hpp"
#include "epf/core/process.hpp"
#include "epf/core/state.hpp"
#include "epf/util/math.hpp"
#include <Eigen/Dense>
#include <random>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/repeat.hpp>
#include <range/v3/view/single.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#include <type_traits>
#include <vector>

namespace epf {

// Put constraint for measurment model in importance sampler first. My original thought during implementing ukf
// importance sampler is:
//
//   std::vector<std::shared_ptr<typename ImportanceSampleStrategy::MeasurmentModel>> sensors_{};
//
// However, I need to make sure how Output (sensor output) interact with the Particle filter, is it ok to combine
// mulitple sensor output to one struct and update partially, is it mathematical doable? if yes, is it doable
// according to current particle filter framework.
//
// For example, if we are going to implement multiple sensor fusion, say, camera, gps, and imu, then our output struct
// will likely be:
//
//    class Output {
//      std::size_t landmark_idx_;
//      double bearing_;
//      double range_;
//      double x_;
//      double y_;
//      std::array<double, 3> omega_;
//      std::array<double, 3> accel_;
//    };
//
// The output vector will be 11 x 1, and particle filter will have three measurement models, during importance
// sampling, we need to calculate cov_yy (for ukf, ekf case). Since not all parts of output vector are updated, some
// entries of cov_yy would likely be 0 (or some meaningless value), what would happen if we take inverse when
// calculating kalman gain?
//
// If we decide that we take variadic Output, then we can only put constraint in importance sampler because we need to
// store the polymorphic type in ParticleFilter class. And the iteration of measurement model will become more
// complicated
//
// Conclusion: I need to make an example for multiple sensor fusion

template <typename State, typename Output>
class UKFSampler {
  double alpha_ = 0.0; /*!< Controls the size of the sigma point. Positive scaling param, can be arbitrarily small to
                          minimize higher order effect */
  double kappa_ = 0.0; /*!< Parameter to guarantee positive semidefiniteness of the covariance matrix */
  double beta_ = 0.0; /*!< Can be used to incorporate knowledge of higher order moments of the distribution. Affects the
                         weighting of the zeroth sigma point for the calculation of the covariance */

  std::mt19937 rng_{std::random_device{}()};

  using StateTrait  = StateTraits<State>;
  using StateNoise  = NoiseTraits<State>;
  using OutputTrait = StateTraits<Output>;
  using OutputNoise = NoiseTraits<Output>;

  static inline constexpr auto OUTPUT_NOISE_DIM    = OutputNoise::Dimension::value;
  static inline constexpr auto OUTPUT_DIM          = OutputTrait::Dimension::value;
  static inline constexpr auto STATE_DIM           = StateTrait::Dimension::value;
  static inline constexpr auto STATE_NOISE_DIM     = StateNoise::Dimension::value;
  static inline constexpr auto AUGMENTED_STATE_DIM = STATE_DIM + STATE_NOISE_DIM + OUTPUT_NOISE_DIM;

  template <std::size_t Row, std::size_t Col>
  using Matrix = Eigen::Matrix<typename StateTrait::ValueType, Row, Col>;

  using AugmentedCovMatrix = Matrix<AUGMENTED_STATE_DIM, AUGMENTED_STATE_DIM>;
  using StateNoiseVector   = typename StateNoise::ArithmeticType;
  using StateVector        = typename StateTrait::ArithmeticType;
  using OutputVector       = typename StateTrait::ArithmeticType;
  using OutputNoiseVector  = typename OutputNoise::ArithmeticType;

  using StateCovariance  = typename StateTrait::CovarianceType;
  using OutputCovariance = typename OutputTrait::CovarianceType;
  using CrossCovMatrix   = Matrix<STATE_DIM, OUTPUT_DIM>;

  std::vector<AugmentedCovMatrix> covariances_;
  std::vector<StateVector> sigma_points_        = std::vector<StateVector>(2 * AUGMENTED_STATE_DIM + 1);
  std::vector<StateNoiseVector> process_noises_ = std::vector<StateNoiseVector>(2 * AUGMENTED_STATE_DIM + 1);
  std::vector<OutputNoiseVector> output_noises_ = std::vector<OutputNoiseVector>(2 * AUGMENTED_STATE_DIM + 1);

  bool initialized_ = false;

 public:
  template <typename T>
  static inline constexpr bool require_meas_model = std::is_base_of_v<MeasurementModel<State, Output>, T>;

  void set_alpha(double const t_param) noexcept { this->alpha_ = t_param; }

  void set_kappa(double const t_param) noexcept { this->kappa_ = t_param; }

  void set_beta(double const t_param) noexcept { this->beta_ = t_param; }

  SamplingResult importance_sampling(ProcessModel<State>* const t_proc, MeasurementModel<State>* const t_meas,
                                     std::vector<StateVector>& t_prev, std::vector<double>& t_weight) {
    using ranges::accumulate;
    using ranges::views::concat;
    using ranges::views::repeat;
    using ranges::views::single;
    using ranges::views::transform;
    using ranges::views::zip;

    auto* const derived = static_cast<MeasurementModel<State, Output>* const>(t_meas);

    constexpr auto Q_begin = STATE_DIM;
    constexpr auto N_X     = AUGMENTED_STATE_DIM;
    constexpr auto R_begin = STATE_DIM + STATE_NOISE_DIM;

    // todo: consider observer pattern to update covariance? thread safety and observer pattern?
    if (not this->initialized_) {
      this->initialized_ = true;
      this->covariances_ = std::vector<AugmentedCovMatrix>(t_prev.size());
      for (auto& cov : this->covariances_) {
        cov.template block<STATE_DIM, STATE_DIM>(0, 0)                           = t_proc->get_process_covariance();
        cov.template block<STATE_NOISE_DIM, STATE_NOISE_DIM>(Q_begin, Q_begin)   = t_proc->get_noise_covariance();
        cov.template block<OUTPUT_NOISE_DIM, OUTPUT_NOISE_DIM>(R_begin, R_begin) = derived->get_noise_covariance();
      }
    }

    double const lambda          = std::pow(this->alpha_, 2) * (N_X + this->kappa_) - N_X;
    double const mean_pt0_weight = lambda / (N_X + lambda);
    double const cov_pt0_weight  = mean_pt0_weight + (1 - std::pow(this->alpha_, 2) + this->beta_);
    double const sigma_pt_weight = 1.0 / (2 * (lambda + N_X));

    auto const mean_weight = concat(single(mean_pt0_weight), repeat(sigma_pt_weight));
    auto const cov_weight  = concat(single(cov_pt0_weight), repeat(sigma_pt_weight));

    for (auto [state, weight, cov] : zip(t_prev, t_weight, this->covariances_)) {
      Eigen::LLT<AugmentedCovMatrix> cholesky_solver{(N_X + lambda) * cov};
      auto const& P_sqrt = cholesky_solver.matrixLLT();

      this->sigma_points_[0] = state;
      for (std::size_t i = 1; i <= N_X; ++i) {
        this->sigma_points_[i]       = state + P_sqrt.template block<STATE_DIM, 1>(0, i - 1);
        this->sigma_points_[N_X + i] = state - P_sqrt.template block<STATE_DIM, 1>(0, i - 1);

        this->process_noises_[i]       = P_sqrt.template block<STATE_NOISE_DIM, 1>(Q_begin, i - 1);
        this->process_noises_[N_X + i] = -this->process_noises_[i];

        this->output_noises_[i]       = P_sqrt.template block<OUTPUT_NOISE_DIM, 1>(R_begin, i - 1);
        this->output_noises_[N_X + i] = -this->output_noises_[i];
      }

      // co_routine or multithread
      t_proc->predict(this->sigma_points_, this->process_noises_);
      StateVector final_mean = StateVector::Zero();
      final_mean             = accumulate(zip(this->sigma_points_, mean_weight), final_mean, weighted_sum);

      auto const to_cov = [](auto const& t_mean) {
        return transform([t_mean](auto const& t_v) {
          auto const mean_diff = t_v - t_mean;
          return mean_diff * mean_diff.transpose();
        });
      };

      StateCovariance cov_xx = StateCovariance::Zero();
      cov_xx = accumulate(zip(this->sigma_points_ | to_cov(final_mean), cov_weight), cov_xx, weighted_sum);

      auto const output_predict      = derived->predict(this->sigma_points_, this->output_noises_);
      OutputVector const output_mean = [&]() {
        OutputVector ret_val = OutputVector::Zero();
        return accumulate(zip(output_predict, mean_weight), ret_val, weighted_sum);
      }();

      auto const cov_yy = [&, cov_yy_tf = output_predict | to_cov(output_mean)]() {
        OutputCovariance ret_val = OutputCovariance::Zero();
        return accumulate(zip(cov_yy_tf, cov_weight), ret_val, weighted_sum);
      }();

      auto const to_cross_cov = [final_mean, output_mean](auto const& t_xy_pair) {
        auto const& [x, y] = t_xy_pair;
        return (x - final_mean) * (y - output_mean).transpose();
      };

      auto const cov_xy = [&, cov_xy_tf = zip(this->sigma_points_, output_predict) | transform(to_cross_cov)]() {
        CrossCovMatrix ret_val = CrossCovMatrix::Zero();
        return accumulate(zip(cov_xy_tf, cov_weight), ret_val, weighted_sum);
      }();

      CrossCovMatrix const k = cov_xy * cov_yy.inverse();
      final_mean += k * (to_state_vector<Output>(derived->get_latest_output()) - output_mean);
      cov_xx -= k * cov_yy * k.transpose();

      MultivariateNormalDistribution<STATE_DIM> dist(final_mean, cov_xx);
      StateVector const sampled  = dist(this->rng_);
      StateVector const avg_diff = sampled - final_mean;
      auto const prob_sampled    = std::exp((-0.5 * avg_diff.transpose() * cov_xx.inverse() * avg_diff)[0]);

      cov.template block<STATE_DIM, STATE_DIM>(0, 0) = cov_xx;

      weight = t_proc->calculate_probability(sampled, state) / prob_sampled;
      state  = sampled;
    }

    // a bit awkward here, because measurement model needs to be careful not to overwrite the weight, i.e., no pure
    // assignment, need *= instead
    t_meas->update(t_prev, t_weight);

    return SamplingResult::Sampled;
  }
};

}  // namespace epf

#endif