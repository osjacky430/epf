#ifndef AMCL_HPP_
#define AMCL_HPP_

#include "kdtree.hpp"
#include "map.hpp"
#include "particle_filter.hpp"
#include "sensor_model.hpp"
#include <boost/math/special_functions.hpp>

namespace amcl {

template <typename ParticleType>
class KLDSampling {
  static constexpr auto DEFAULT_EPSILON        = 0.01;
  static constexpr auto DEFAULT_UPPER_QUANTILE = 3;

  ParticleType bin_size_;  // shouldn't be particle size, eigen::array<x>
  double epsilon_;         /* !< Maximum allowed error (K-L distance) between true and estimated distribution */

  // according to standard normal distribution wikipedia. A normal random variable X will exceed mu + z_{1-p} * sigma
  // with probability p (i.e. within mu + z_{1-p} with probability 1 - p), and will lie outside the interval [mu -
  // z_{1-p} * sigma, mu + z_{1-p} * sigma] with the probability 2 * p. (p is changed to 1 - p to match the notation
  // used in the paper)
  double upper_quantile_; /* !< z_{1-p}, upper quantile of standard normal distribution N(0, 1). where 1 - p is the
                      probability that K-L distance between MLE and the true distribution is smaller than epsilon */

  std::size_t min_particle_count_;
  std::size_t max_particle_count_;

  /* !< Data structure to store histogram */
  particle_filter::KDTree<ParticleType, typename ParticleType::PivotComp> kdtree_;

 public:
  [[nodiscard]] std::size_t max_particle() const noexcept { return this->max_particle_count_; }

  [[nodiscard]] std::size_t min_particle() const noexcept { return this->min_particle_count_; }

  // kd tree range search?
  void cluster_particles() {}

  /**
   *  @brief  This function inserts particle, divided by resolution (a.k.a bin size in our
   *          application), to kdtree, then calculate particle size accordingly
   *
   *  @note   According to [1], the bin size is better fixed at 50 (cm) * 50 (cm) * 10 (deg) for 2D case
   *
   *  [1] Fox, Dieter. "Adapting the Sample Size in Particle Filters Through KLD-Sampling"
   */
  [[nodiscard]] std::size_t calculate_particle_size(std::vector<ParticleType> const& t_particles) noexcept {
    this->kdtree_.clear();

    std::for_each(t_particles.begin(), t_particles.end(),
                  [this](auto const& t_particle) { this->kdtree_.insert(t_particle / this->bin_size_); });

    if (this->kdtree_.size() <= 1) {
      return this->min_particle_count_;
    }

    auto const k = static_cast<double>(this->kdtree_.size());
    auto const x = 1.0 - 2.0 / (9.0 * (k - 1.0)) + std::sqrt(2.0 / (9.0 * (k - 1.0))) * this->upper_quantile_;
    auto const adaptive_particle_count = std::ceil(x * x * x * (k - 1) / (2 * this->epsilon_));

    return static_cast<std::size_t>(std::clamp(adaptive_particle_count, static_cast<double>(this->min_particle_count_),
                                               static_cast<double>(this->max_particle_count_)));
  }

  KLDSampling() = default;
  explicit KLDSampling(std::size_t const t_min_particle_count, std::size_t const t_max_particle_count,
                       ParticleType t_bin_size, double const t_epsilon = DEFAULT_EPSILON,
                       double const t_upper_quantile = DEFAULT_UPPER_QUANTILE) noexcept
    : bin_size_(t_bin_size),
      epsilon_(t_epsilon),
      upper_quantile_(t_upper_quantile),
      min_particle_count_(t_min_particle_count),
      max_particle_count_(t_max_particle_count) {}

  /**
   *  @brief
   */
  static double quantile_from_p(double const t_prob) noexcept {
    assert(0 < t_prob and t_prob < 1);
    return std::sqrt(2) * boost::math::erf_inv(2 * t_prob - 1);
  }
};

// augmented, with w_slow and w_fast to solve robot kidnapping
template <typename ParticleType, template <typename T> typename ParticleSizeStrategy = KLDSampling>
class AMCLResample : ParticleSizeStrategy<ParticleType> {
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

  explicit AMCLResample(double const t_alpha_slow = DEFAULT_ALPHA_SLOW,
                        double const t_alpha_fast = DEFAULT_ALPHA_FAST) noexcept
    : alpha_slow_(t_alpha_slow), alpha_fast_(t_alpha_fast) {}

  std::vector<ParticleType> resample(particle_filter::SensorModel<ParticleType>* const t_sensor,
                                     std::vector<ParticleType> const& t_previous_particles,
                                     particle_filter::MapBase<ParticleType> const& t_map) noexcept {
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
        // Generation of random particle depends on sensor characteristics. For laser ranger, we can generate random
        // particle by certain distribution, then assign weight to them according to current observation; for landmark
        // detection, however, this isn't suitable (if the robot can see landmark located at point A, then generate a
        // random point that can't see the landmark makes no sense, also, it is difficult to determine whether the
        // robot can see the landmark at a random point or not)

        particle = t_sensor->sample_pose_from_latest_measurement(t_map);
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

    std::for_each(ret_val.begin(), ret_val.end(), [&](auto& t_particle) { t_particle /= total_weight; });

    return ret_val;
  }
};

template <typename ParticleType, typename Resampler = AMCLResample<ParticleType>>
class AMCL2D final : public particle_filter::ParticleFilter<ParticleType, Resampler> {
 private:
  std::vector<ParticleType> previous_particles_{};

  using PF = particle_filter::ParticleFilter<ParticleType, Resampler>;

 public:
  explicit AMCL2D(std::size_t t_max_particles = PF::DEFAULT_MAX_PARTICLE_COUNTS)
    : previous_particles_(t_max_particles, ParticleType({0, 0, 0}, 1.0 / static_cast<double>(t_max_particles))) {}

  ParticleType sample(particle_filter::MapBase<ParticleType>& t_map) noexcept override {
    for (auto& sensor : this->get_all_sensors()) {
      // When the robot stopped, resampling should be suspended (and in fact it is usually a good idea to suspend the
      // integration of measurements as well). If we clearly know that the robot doesn't move at all, then the diversity
      // of the particles should remain the same (what remains unknown should still be unknown), resampling process
      // induces the loss of diversity, even though the variance of particle set decreases, the variance of the particle
      // set as an estimator of TRUE belief increases.
      if (auto status = this->get_motion_model()->update_motion(this->previous_particles_);
          status == particle_filter::UpdateStatus::NoUpdate) {
        continue;
      }

      auto const estimate_result = sensor->estimate(this->previous_particles_, t_map);
      if (estimate_result == particle_filter::SensorResult::NoMeasurement) {  // no valid estimation yet
        continue;
      }

      this->previous_particles_ = this->resample(sensor.get(), this->previous_particles_, t_map);
    }

    auto const weighted_sum = [this](auto const& t_left, auto const& t_right) {
      return t_left + t_right * t_right.weight();
    };

    return std::accumulate(this->previous_particles_.begin(), this->previous_particles_.end(), ParticleType(),
                           weighted_sum);
  }
};

}  // namespace amcl

#endif