#ifndef OMNIDIRECTIONAL_DRIVE_HPP_
#define OMNIDIRECTIONAL_DRIVE_HPP_

#include "epf/component/concept/pose.hpp"
#include "epf/core/measurement.hpp"
#include "epf/core/process.hpp"
#include "epf/util/traits.hpp"
#include <range/v3/algorithm/for_each.hpp>

namespace epf {

// extract common interface? something like drive_model.hpp?
template <typename PoseData, typename State>
class OmniDirectionalDrive final : public ProcessModel<State> {
 public:
  using StateVector  = typename StateTraits<State>::ArithmeticType;
  using PoseDataDiff = PoseData;

  OmniDirectionalDrive() = default;

  void attach_measurement(Measurement<PoseData>* t_odom_meas) noexcept { this->odom_meas_ = t_odom_meas; }

  void set_update_threshold(PoseDataDiff const& t_diff) noexcept { this->threshold_ = t_diff; }

  void set_alpha_strafe(std::array<double, 2> const& t_alpha_strafe) noexcept { this->alpha_strafe_ = t_alpha_strafe; }
  void set_alpha_rot(std::array<double, 2> const& t_alpha_rot) noexcept { this->alpha_rot_ = t_alpha_rot; }
  void set_alpha_trans(std::array<double, 2> const& t_alpha_trans) noexcept { this->alpha_trans_ = t_alpha_trans; }

  [[nodiscard]] bool input_ready() override { return this->odom_meas_->data_ready(); }

  [[nodiscard]] Prediction predict(std::vector<StateVector>& t_particles) override {
    auto const current_measurement = this->odom_meas_->get_measurement();

    bool const valid_threshold = this->threshold_[0] > 0.0 and this->threshold_[1] > 0.0 and this->threshold_[2] > 0.0;
    auto const diff_x          = current_measurement[0] - this->previous_[0];
    auto const diff_y          = current_measurement[1] - this->previous_[1];
    auto const diff_w          = constraint_angle(current_measurement[2] - this->previous_[2]);
    if (valid_threshold and (std::abs(diff_x) < this->threshold_[0] and std::abs(diff_y) < this->threshold_[1] and
                             std::abs(diff_w) < this->threshold_[2])) {
      return Prediction::NoUpdate;
    }

    double const dtrans = std::sqrt(diff_x * diff_x + diff_y * diff_y);
    double const drot   = diff_w;

    double const dtrans_sqr = std::pow(dtrans, 2);
    double const drot_sqr   = std::pow(drot, 2);

    double const var_trans  = this->alpha_trans_[0] * dtrans_sqr + this->alpha_trans_[1] * drot_sqr;
    double const var_rot    = this->alpha_rot_[0] * dtrans_sqr + this->alpha_rot_[1] * drot_sqr;
    double const var_strafe = this->alpha_strafe_[0] * dtrans_sqr + this->alpha_strafe_[1] * drot_sqr;

    auto rot_dist    = std::normal_distribution<>(0, std::sqrt(var_rot));
    auto strafe_dist = std::normal_distribution<>(0, std::sqrt(var_strafe));
    auto trans_dist  = std::normal_distribution<>(0, std::sqrt(var_trans));

    auto const esitmate_per_particle = [&](auto& t_current) {
      double const drot_hat    = constraint_angle(drot - rot_dist1(this->rng_));
      double const dtrans_hat  = dtrans - trans_dist(this->rng_);
      double const dstrafe_hat = strafe_dist(this->rng_);

      // ros-navigation use bearing angle = std::atan(diff_y, diff_x) + w_coor<State>(t_current) - this->previous_[2]
      // still having hard time understanding it... (makes no sense to me, I thought w_coor<State>(t_current) and
      // this->previous_[2] are almost the same, if I'm correct, then why do this. If these two are very different, what
      // does the subtraction even mean?)
      double const move_dir_angle = std::atan2(diff_y, diff_x);
      x_coor<State>(t_current) += (dtrans_hat * std::cos(move_dir_angle) + dstrafe_hat * std::sin(move_dir_angle));
      y_coor<State>(t_current) += (dtrans_hat * std::sin(move_dir_angle) - dstrafe_hat * std::sin(move_dir_angle));
      w_coor<State>(t_current) += drot_hat;
      w_coor<State>(t_current) = constraint_angle(w_coor<State>(t_current));
    };

    ranges::for_each(t_particles, esitmate_per_particle);
    this->previous_ = current_measurement;

    return Prediction::Updated;
  }

 private:
  // some thought: why enforce subscript operator instead of x_coor<> specialization
  static_assert(has_subscript_operator<PoseData>::value);
  BOOST_CONCEPT_ASSERT((Pose2DConcept<State>));

  std::array<double, 2> alpha_trans_{0, 0};
  std::array<double, 2> alpha_rot_{0, 0};
  std::array<double, 2> alpha_strafe_{0, 0};

  Measurement<PoseData>* odom_meas_ = nullptr;
  std::mt19937 rng_{std::random_device()()};

  PoseData previous_{};
  PoseDataDiff threshold_{};
};

}  // namespace epf

#endif