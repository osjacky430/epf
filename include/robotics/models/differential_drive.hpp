#ifndef DIFFERENTIAL_DRIVE_HPP_
#define DIFFERENTIAL_DRIVE_HPP_

#include "core/process.hpp"

namespace epf {

template <typename ParticleType>
struct Differential final : public ProcessModel<ParticleType> {
  std::mt19937 rng_ = std::mt19937(std::random_device()());

  ParticleType previous_{};
  std::array<double, 2> alhpa_rot_   = {0, 0};
  std::array<double, 2> alhpa_trans_ = {0, 0};

  // differential drive model is aware of 2D space only, therefore it is safe to assume
  // particle type can be accessed via index operator, with size = 3 (x, y, yaw angle)
  [[nodiscard]] Prediction predict(std::vector<ParticleType>& t_particles) override {
    ParticleType current_odom_meas{};  // its motion model's responsibility to get odom measurement
    auto const diff = current_odom_meas - this->previous_;
    if (diff[0] < 1e-6) {  //  some threshold, make it a user adjustable param
      return Prediction::NoUpdate;
    }

    double const delta_rot1  = std::atan2(diff[1], diff[0]) - this->previous_[2];
    double const delta_rot2  = diff[2] - delta_rot1;
    double const delta_trans = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1]);

    // TODO: amcl use square instead of abs, check why
    double const variance_rot   = this->alhpa_rot_[0] * delta_rot1 + this->alhpa_rot_[1] * delta_trans;
    double const variance_trans = this->alhpa_trans_[0] * delta_trans + this->alhpa_trans_[1] * diff[2];

    auto rot_dist   = std::normal_distribution<>(0, variance_rot);
    auto trans_dist = std::normal_distribution<>(0, variance_trans);

    auto const esitmate_per_particle = [&](auto& t_current) {
      double const delta_rot1_hat  = delta_rot1 - rot_dist(this->rng_);
      double const delta_trans_hat = delta_trans - trans_dist(this->rng_);
      double const delta_rot2_hat  = delta_rot2 - rot_dist(this->rng_);

      t_current += ParticleType{{
        delta_trans_hat * std::cos(t_current[2] + delta_rot1_hat),
        delta_trans_hat * std::cos(t_current[2] + delta_rot1_hat),
        delta_rot1_hat + delta_rot2_hat,
      }};
    };

    std::for_each(t_particles.begin(), t_particles.end(), esitmate_per_particle);
    this->previous_ = current_odom_meas;

    return Prediction::Updated;
  }
};

}  // namespace epf

#endif