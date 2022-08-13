#ifndef ADAPTIVE_RESAMPLER_HPP_
#define ADAPTIVE_RESAMPLER_HPP_

#include "core/measurement.hpp"
#include "map.hpp"  // TODO: remove

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

namespace epf {

// augmented, with w_slow and w_fast to solve robot kidnapping
template <typename ParticleType, template <typename T> typename ParticleSizeStrategy>
class AdaptiveResample : ParticleSizeStrategy<ParticleType> {
  double w_slow_ = 0.0; /*!< long term average */
  double w_fast_ = 0.0; /*!< short term average */

  std::mt19937 rng_ = std::mt19937(std::random_device{}());

  double alpha_slow_ = DEFAULT_ALPHA_SLOW; /*!< cut off frequency (?) for long term average */
  double alpha_fast_ = DEFAULT_ALPHA_FAST; /*!< cut off frequency (?) for short term average */

 public:
  using ParticleSizeStrategy<ParticleType>::calculate_particle_size;

  // The algorithm requires alph_slow << alpha_fast
  static constexpr auto DEFAULT_ALPHA_SLOW = 0.001;
  static constexpr auto DEFAULT_ALPHA_FAST = 0.1;

  explicit AdaptiveResample(double const t_alpha_slow = DEFAULT_ALPHA_SLOW,
                            double const t_alpha_fast = DEFAULT_ALPHA_FAST) noexcept
    : alpha_slow_(t_alpha_slow), alpha_fast_(t_alpha_fast) {
    assert(t_alpha_fast / t_alpha_slow >= 100.0);
  }

  void set_alpha_slow(double const t_new_value) noexcept { this->alpha_slow_ = t_new_value; }
  void set_alpha_fast(double const t_new_value) noexcept { this->alpha_fast_ = t_new_value; }

  std::vector<ParticleType> resample(epf::MeasurementModel<ParticleType>* const t_meas_model,
                                     std::vector<ParticleType> const& t_previous_particles) noexcept {
    auto const sum_weight = [](auto&& t_value, auto&& t_particles) { return t_value + t_particles.weight(); };
    auto const w_sum      = std::accumulate(t_previous_particles.begin(), t_previous_particles.end(), 0.0, sum_weight);
    auto const w_avg      = w_sum / static_cast<double>(t_previous_particles.size());

    this->w_slow_ += this->alpha_slow_ * (w_avg - this->w_slow_);
    this->w_fast_ += this->alpha_fast_ * (w_avg - this->w_fast_);

    auto const cumulative_weight = [&]() {
      std::vector<double> ret_val(t_previous_particles.size() + 1, 0);
      for (std::size_t i = 0; i < t_previous_particles.size(); ++i) {
        ret_val[i + 1] = t_previous_particles[i].weight() + ret_val[i];
      }

      return ret_val;
    }();

    std::vector<ParticleType> ret_val(t_previous_particles.size());
    if (cumulative_weight.back() == 0.0) {
      std::transform(t_previous_particles.begin(), t_previous_particles.end(), ret_val.begin(), [&](auto t_particle) {
        t_particle.weight() = 1.0 / static_cast<double>(t_previous_particles.size());
        return t_particle;
      });
      return ret_val;
    }

    double total_weight = 0;
    for (auto const w_diff = std::max(0.0, 1.0 - this->w_fast_ / this->w_slow_);
         ret_val.size() < this->max_particle();) {
      ParticleType particle;
      std::uniform_real_distribution uniform_dist(0.0);
      if (auto const rand_val = uniform_dist(this->rng_); rand_val < w_diff) {
        // Generation of random particle depends on sensor characteristics. For laser range finder, we can generate
        // random particle by certain distribution, then assign weight to them according to current observation; for
        // landmark detection, however, this isn't suitable (if the robot can see landmark located at point A, then
        // generate a random point that can't see the landmark makes no sense, also, it is difficult to determine
        // whether the robot can see the landmark at a random point or not)

        particle = t_meas_model->sample_state_from_latest_measurement();
      } else {
        // sample from previous particles. We use std::lower_bound since cumulative weight is sorted by nature
        // (strictly increasing, as probability won't be zero (?)), even tho using binary search on floating point
        // might be a bit shaky (?)
        auto const iter = std::lower_bound(cumulative_weight.begin(), cumulative_weight.end(), rand_val);
        particle        = t_previous_particles[static_cast<std::size_t>(iter - cumulative_weight.begin())];
      }

      total_weight += particle.weight();

      if (ret_val.size() >= this->calculate_particle_size(t_previous_particles)) {
        break;
      }
    }

    std::for_each(ret_val.begin(), ret_val.end(), [&](auto& t_particle) { t_particle.weight() /= total_weight; });

    return ret_val;
  }
};

}  // namespace epf

#endif