#ifndef DIFFERENTIAL_DRIVE_HPP_
#define DIFFERENTIAL_DRIVE_HPP_

#include "epf/core/measurement.hpp"
#include "epf/core/process.hpp"
#include "epf/core/state.hpp"
#include "epf/util/math.hpp"
#include "epf/util/traits.hpp"
#include <boost/concept_check.hpp>
#include <boost/math/constants/constants.hpp>

namespace epf {

template <typename PoseData, typename PoseDataDiff>
struct DifferentialDriveParam {
  std::array<double, 2> alpha_rot_;
  std::array<double, 2> alpha_trans_;
  PoseData initial_pose_;
  PoseDataDiff threshold_;
};

// temporary, not sure if this is the best api
template <typename State>
struct DifferentialModelStateConcept {  // NOLINT(*-special-member-functions), because this class is not meant to be
                                        // instantiated
  BOOST_CONCEPT_USAGE(DifferentialModelStateConcept) {
    using StateValue  = typename StateTraits<State>::ValueType;
    using StateVector = typename StateTraits<State>::ArithmeticType;

    StateVector s{};
    [[maybe_unused]] StateValue& ref_x = x_coor<State>(s);
    [[maybe_unused]] StateValue& ref_y = y_coor<State>(s);
    [[maybe_unused]] StateValue& ret_w = w_coor<State>(s);
  }
};

/**
 *  @brief  This class is a concrete implementation of differential drive process model on flat surface
 *
 *  @tparam PoseData  Data structure for odometry data, required to implement subscript operator, representing [x,
 *                    y, w]. (coordinate, not displacement)
 *  @tparam State Data structure for state estimation, required to implement function x(), y(), and w(), that return
 *                reference to [x, y, w] variable.
 *
 */
template <typename PoseData, typename State>
class Differential final : public ProcessModel<State> {
  using PoseDataDiff = PoseData;
  using StateVector  = typename StateTraits<State>::ArithmeticType;

  Measurement<PoseData>* odom_meas_ = nullptr;
  std::mt19937 rng_                 = std::mt19937(std::random_device()());

  PoseData previous_{};
  PoseDataDiff threshold_{};

  std::array<double, 2> alpha_rot_   = {0, 0};
  std::array<double, 2> alpha_trans_ = {0, 0};

  static_assert(has_subscript_operator<PoseData>::value);
  BOOST_CONCEPT_ASSERT((DifferentialModelStateConcept<State>));

 public:
  Differential() = default;
  Differential(DifferentialDriveParam<PoseData, PoseDataDiff> const& t_param, Measurement<PoseData>* const t_odom_meas)
    : alpha_rot_(t_param.alpha_rot_),
      alpha_trans_(t_param.alpha_trans_),
      previous_{t_param.initial_pose_},
      threshold_(t_param.threshold_),
      odom_meas_(t_odom_meas) {}

  void set_pose(PoseData const& t_pose) noexcept { this->previous_ = t_pose; }
  void set_alpha_rot(std::array<double, 2> const& t_alpha_rot) noexcept { this->alpha_rot_ = t_alpha_rot; }
  void set_alpha_trans(std::array<double, 2> const& t_alpha_trans) noexcept { this->alpha_trans_ = t_alpha_trans; }

  void set_update_threshold(PoseDataDiff const& t_diff) noexcept { this->threshold_ = t_diff; }

  void attach_measurement(Measurement<PoseData>* t_odom_meas) noexcept { this->odom_meas_ = t_odom_meas; }

  [[nodiscard]] bool input_ready() override { return this->odom_meas_->data_ready(); }

  [[nodiscard]] Prediction predict(std::vector<StateVector>& t_particles) override {
    auto const current_measurement = this->odom_meas_->get_measurement();

    // if we do not have valid threshold, then we assume that user wants to update every single time
    // this part can be moved to input_ready
    bool const valid_threshold = this->threshold_[0] > 0.0 and this->threshold_[1] > 0.0 and this->threshold_[2] > 0.0;
    auto const diff_x          = current_measurement[0] - this->previous_[0];
    auto const diff_y          = current_measurement[1] - this->previous_[1];
    auto const diff_w          = constraint_angle(current_measurement[2] - this->previous_[2]);
    if (valid_threshold and (std::abs(diff_x) < this->threshold_[0] and std::abs(diff_y) < this->threshold_[1] and
                             std::abs(diff_w) < this->threshold_[2])) {
      return Prediction::NoUpdate;
    }

    double const delta_trans = std::sqrt(diff_x * diff_x + diff_y * diff_y);
    auto const delta_rot_tup = [&]() {
      // according to the derivation, delta_rot1 should be 0 for in-place rotation situation. We need to treat it as
      // special case because diff_x and diff_y are too small to get meaningful value by atan2
      if (delta_trans <= std::hypot(this->threshold_[0], this->threshold_[1])) {
        return std::pair{0.0, 1.0};
      }

      // original model assumes that the robot will turn to move_direction, move to destination, then rotate to final
      // orientation. However, we don't necessary need to turn to move_direction, we could also turn to its complement
      // angle, i.e. pi - move_direction, and move "backward" toward destination. This may improve the modelling
      // accuracy since the more unnecessary moves we did, the more noise comes into play (due to bigger var_rot1 and
      // var_rot2). Also, we normally choose the most efficient way to move the robot, so this should be more intuitive.
      // Notice that we don't need to follow the actual velocity cmd, since this model is based on odometry measurement,
      // instead of velocity. Therefore any assumption should work, as long as the odometry diff is satisfied, the only
      // difference is the distribution of propagated particles.
      auto const move_direction = constraint_angle(std::atan2(diff_y, diff_x) - this->previous_[2]);
      if (std::abs(move_direction) > std::abs(boost::math::double_constants::pi - move_direction)) {
        return std::pair{boost::math::double_constants::pi - move_direction, -1.0};
      }
      return std::pair{constraint_angle(move_direction), 1.0};
    }();

    double const delta_rot1 = std::get<0>(delta_rot_tup);
    double const rot1_sign  = std::get<1>(delta_rot_tup);
    double const delta_rot2 = constraint_angle(diff_w - delta_rot1);

    // ros-navigation calculate variance differently, and that is totally fine, because these variances can be tuned
    // with alpha1 ~ alpha4, different ways of calculation simply means different suitable parameters.
    double const var_rot1  = this->alpha_rot_[0] * std::abs(delta_rot1) + this->alpha_rot_[1] * std::abs(delta_trans);
    double const var_rot2  = this->alpha_rot_[0] * std::abs(delta_rot2) + this->alpha_rot_[1] * std::abs(delta_trans);
    double const var_trans = this->alpha_trans_[0] * std::abs(delta_trans) + this->alpha_trans_[1] * std::abs(diff_w);

    auto rot_dist1  = std::normal_distribution<>(0, std::sqrt(var_rot1));
    auto rot_dist2  = std::normal_distribution<>(0, std::sqrt(var_rot2));
    auto trans_dist = std::normal_distribution<>(0, std::sqrt(var_trans));

    auto const esitmate_per_particle = [&](auto& t_current) {
      double const delta_rot1_hat  = constraint_angle(delta_rot1 - rot_dist1(this->rng_));
      double const delta_trans_hat = std::abs(delta_trans - trans_dist(this->rng_));
      double const delta_rot2_hat  = constraint_angle(delta_rot2 - rot_dist2(this->rng_));

      double const move_dir_angle = constraint_angle(w_coor<State>(t_current) + delta_rot1_hat);
      x_coor<State>(t_current) += rot1_sign * delta_trans_hat * std::cos(move_dir_angle);
      y_coor<State>(t_current) += rot1_sign * delta_trans_hat * std::sin(move_dir_angle);
      w_coor<State>(t_current) += (delta_rot1_hat + delta_rot2_hat);
      w_coor<State>(t_current) = constraint_angle(w_coor<State>(t_current));
    };

    std::for_each(t_particles.begin(), t_particles.end(), esitmate_per_particle);
    this->previous_ = current_measurement;

    return Prediction::Updated;
  }
};

}  // namespace epf

#endif