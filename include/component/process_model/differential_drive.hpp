#ifndef DIFFERENTIAL_DRIVE_HPP_
#define DIFFERENTIAL_DRIVE_HPP_

#include "core/measurement.hpp"
#include "core/process.hpp"
#include "util/traits.hpp"
#include <boost/concept_check.hpp>

namespace epf {

template <typename State>
class DifferentialModelStateConcept {
  BOOST_CONCEPT_USAGE(DifferentialModelStateConcept) {
    auto p = State{};

    [[maybe_unused]] double& ref_x = p.x();
    [[maybe_unused]] double& ref_y = p.y();
    [[maybe_unused]] double& ret_w = p.w();
  }
};

/**
 *  @brief  This class is a concrete implementation of differential drive process model on flat surface
 *
 *  @tparam OdomData  Data structure for odometry data, required to implement subscript operator, representing [x,
 *                    y, w]. (coordinate, not displacement)
 *  @tparam State Data structure for state estimation, required to implement function x(), y(), and w(), that return
 *                reference to [x, y, w] variable.
 *
 */
template <typename OdomData, typename State>
class Differential final : public ProcessModel<State> {
  using OdomDataDiff = OdomData;

  Measurement<OdomData>* odom_meas_ = nullptr;
  std::mt19937 rng_                 = std::mt19937(std::random_device()());

  OdomData previous_{};
  OdomDataDiff threshold_{};

  std::array<double, 2> alhpa_rot_   = {0, 0};
  std::array<double, 2> alhpa_trans_ = {0, 0};

  static_assert(has_subscript_operator<OdomData>::value);
  // BOOST_CONCEPT_ASSERT(DifferentialModelStateConcept<State>);

 public:
  void set_alpha_rot(std::array<double, 2> const& t_alpha_rot) noexcept { this->alhpa_rot_ = t_alpha_rot; }
  void set_trans_rot(std::array<double, 2> const& t_alpha_trans) noexcept { this->alhpa_trans_ = t_alpha_trans; }

  void set_update_threshold(OdomDataDiff const& t_diff) noexcept { this->threshold_ = t_diff; }

  void attach_measurement(Measurement<OdomData>* t_odom_meas) noexcept { this->odom_meas_ = t_odom_meas; }

  [[nodiscard]] Prediction predict(std::vector<State>& t_particles) override {
    auto result = this->odom_meas_->get_measurement();
    if (result == std::nullopt) {
      return Prediction::NoUpdate;
    }

    auto const current_measurement = *result;

    auto const diff_x = current_measurement[0] - this->previous_[0];
    auto const diff_y = current_measurement[1] - this->previous_[1];
    auto const diff_w = current_measurement[2] - this->previous_[2];
    if (std::abs(diff_x) < this->threshold_[0] and std::abs(diff_y) < this->threshold_[1] and
        std::abs(diff_w) < this->threshold_[2]) {
      return Prediction::NoUpdate;
    }

    double const delta_rot1  = std::atan2(diff_y, diff_x) - this->previous_[2];
    double const delta_rot2  = diff_y - delta_rot1;
    double const delta_trans = std::sqrt(diff_x * diff_x + diff_y * diff_y);

    // TODO: amcl use square instead of abs, check why
    double const variance_rot   = this->alhpa_rot_[0] * delta_rot1 + this->alhpa_rot_[1] * delta_trans;
    double const variance_trans = this->alhpa_trans_[0] * delta_trans + this->alhpa_trans_[1] * diff_w;

    auto rot_dist   = std::normal_distribution<>(0, variance_rot);
    auto trans_dist = std::normal_distribution<>(0, variance_trans);

    auto const esitmate_per_particle = [&](auto& t_current) {
      double const delta_rot1_hat  = delta_rot1 - rot_dist(this->rng_);
      double const delta_trans_hat = delta_trans - trans_dist(this->rng_);
      double const delta_rot2_hat  = delta_rot2 - rot_dist(this->rng_);

      t_current.x() += delta_trans_hat * std::cos(t_current.x() + delta_rot1_hat);
      t_current.y() += delta_trans_hat * std::cos(t_current.y() + delta_rot1_hat);
      t_current.w() += (delta_rot1_hat + delta_rot2_hat);
    };

    std::for_each(t_particles.begin(), t_particles.end(), esitmate_per_particle);
    this->previous_ = current_measurement;

    return Prediction::Updated;
  }
};

}  // namespace epf

#endif