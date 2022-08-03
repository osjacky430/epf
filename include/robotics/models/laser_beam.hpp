#ifndef LASER_BEAM_HPP_
#define LASER_BEAM_HPP_

#include "sensor_model.hpp"

#include <boost/math/constants/constants.hpp>
#include <random>

namespace particle_filter {

/**
 *  @brief  Concrete implementation of sensor_model, modelling laser range finders (2D)
 */
template <typename MeasurementType, typename ParticleType>
class LaserBeamModel final : public SensorModel<ParticleType> {
  MeasurementType latest_measurement_;
  std::mt19937 rng_ = std::mt19937(std::random_device{}());
  typename ParticleType::ValueType laser_pose_{}; /*!< Laser position relative to the robot, we assumed that
                                                     ParticleType::ValueType has the meaning of "pose" */
  double max_range_ = 0.0;

  double z_hit_     = 0.0; /*!< Correct range with local measurement noise */
  double sigma_hit_ = 0.0; /*!< This noise is usually modeled by a narrow Gaussian with mean and standard deviation
                                sigma , an intrinsic noise parameter of the measurement model */

  double z_short_      = 0.0; /*!< Unexpected objects */
  double lambda_short_ = 0.0; /*!< the likelihood of sensing unexpected objects decreases with range, modeled by
                                   exponential distribution */

  double z_max_  = 0.0; /*!< Failures */
  double z_rand_ = 0.0; /*!< Random measurements*/

 public:
  /**
   *  @brief
   *
   */
  SensorResult estimate(std::vector<ParticleType>& t_particles,
                        particle_filter::MapBase<ParticleType>& t_map) override {
    // future, promise?
    MeasurementType income_meas; /* it is sensor's reponsibility to obtain its measurement, and to determine whether to
                                    update or not */
    double weight_sum = 0.0;
    for (auto& particle : t_particles) {
      double estimated_weight = 1.0;
      // doing so enforce:
      //    1. ParticleType implement value() that returns ParticleType::ValueType
      //    2. the return type of value() should implement method
      //       ParticleType::ValueType transform_coordinate(ParticleType::ValueType)
      auto const laser_pose = particle.value().transform_coordinate(this->laser_pose_);  // find laser position in map

      auto const estimate_per_beam = [&](auto const& t_laser_data) {
        // TODO: generate unit vector from rpy or quaternion (add constraint to MeasurementType)
        std::array<double, 3> const unit_vector = {std::cos(t_laser_data.second), std::sin(t_laser_data.second), 0};
        auto const theoretical_range            = t_map.distance_to_obstacle(laser_pose, this->max_range_, unit_vector);

        auto const diff = t_laser_data.first - theoretical_range;

        // TODO: investigate why no normalization needed
        auto const p_hit   = std::exp(-(diff * diff) / 2.0 * this->sigma_hit_ * this->sigma_hit_);
        auto const p_short = this->lambda_short_ * std::exp(-this->lambda_short_ * t_laser_data.first);
        auto const p_max   = static_cast<int>(t_laser_data.first == this->max_range_);
        auto const p_rand  = 1.0 / this->max_range_;

        // importance factor for this measurement
        auto const p = (this->z_hit_ * p_hit + this->z_short_ * p_short  //
                        + this->z_max_ * p_max + this->z_rand_ * p_rand);
        // TODO: amcl use estimated_weight += p * p * p, not sure why
        estimated_weight *= p;
      };

      // TODO: do beam skip
      std::for_each(income_meas.raw_measurements_.begin(), income_meas.raw_measurements_.end(), estimate_per_beam);

      // TODO: amcl use particle.weight *= estimated_weight, not sure why
      particle.weight() = estimated_weight;
      weight_sum += particle.weight();
    }

    this->latest_measurement_ = income_meas;
    return SensorResult::Estimated;
  }

  /**
   *
   */
  ParticleType sample_pose_from_latest_measurement(MapBase<ParticleType> const& t_map) override {
    static constexpr auto pi = boost::math::constants::pi<double>();

    // this is wrong (because it assumes 2d case)
    auto const& free_cells = t_map.get_free_cells();
    auto const rng_idx     = std::uniform_int_distribution<std::size_t>(0, free_cells.size())(this->rng_);
    auto const free_cell   = free_cells[rng_idx];
    auto const rng_angle   = std::uniform_real_distribution<>(-pi, pi)(this->rng_);
    ParticleType particle{{free_cell[0], free_cell[1], rng_angle}};

    auto const laser_pose = particle.value().transform_coordinate(this->laser_pose_);  // find laser position in map
    auto const estimate_per_beam = [&](auto const& t_laser_data) {
      auto const unit_vector       = std::array{std::cos(t_laser_data.second), std::sin(t_laser_data.first), 0.0};
      auto const theoretical_range = t_map.distance_to_obstacle(laser_pose, this->max_range_, unit_vector);

      auto const diff = t_laser_data.first - theoretical_range;

      // TODO: investigate why no normalization needed
      auto const p_hit   = std::exp(-(diff * diff) / 2.0 * this->sigma_hit_ * this->sigma_hit_);
      auto const p_short = this->lambda_short_ * std::exp(-this->lambda_short_ * t_laser_data.first);
      auto const p_max   = static_cast<int>(t_laser_data.first == this->max_range_);
      auto const p_rand  = 1.0 / this->max_range_;

      // importance factor for this measurement
      auto const p = (this->z_hit_ * p_hit + this->z_short_ * p_short  //
                      + this->z_max_ * p_max + this->z_rand_ * p_rand);
      // TODO: amcl use estimated_weight += p * p * p, not sure why
      particle.weight() *= p;
    };

    std::for_each(this->latest_measurement_.raw_measurements_.begin(),
                  this->latest_measurement_.raw_measurements_.end(), estimate_per_beam);

    return particle;
  }
};

}  // namespace particle_filter

#endif