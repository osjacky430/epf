#ifndef ADAPTIVE_SAMPLE_SIZE_HPP_
#define ADAPTIVE_SAMPLE_SIZE_HPP_

#include "core/kdtree.hpp"
#include "util/traits.hpp"

#include <algorithm>
#include <boost/math/special_functions.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>

namespace epf {

template <typename ParticleType>
class KLDSampling {
  ParticleType bin_size_;  // shouldn't be particle size, eigen::array<x>

  static constexpr auto DEFAULT_EPSILON = 0.01;

  /* !< Maximum allowed error (K-L distance) between true and estimated distribution */
  double epsilon_ = DEFAULT_EPSILON;

  // according to standard normal distribution wikipedia. A normal random variable X will exceed mu + z_{1-p} * sigma
  // with probability p (i.e. within mu + z_{1-p} with probability 1 - p), and will lie outside the interval [mu -
  // z_{1-p} * sigma, mu + z_{1-p} * sigma] with the probability 2 * p. (p is changed to 1 - p to match the notation
  // used in the paper), default upper quantile = 3 corresponds to p = 0.001, i.e. 1 - p = 0.999
  static constexpr auto DEFAULT_UPPER_QUANTILE = 3;

  /* !< z_{1-p}, upper quantile of standard normal distribution N(0, 1). where 1 - p is the probability that K-L
   *    distance between MLE and the true distribution is smaller than epsilon */
  double upper_quantile_ = DEFAULT_UPPER_QUANTILE;

  std::size_t min_particle_count_{};
  std::size_t max_particle_count_{};

  /* !< Data structure to store histogram */
  epf::KDTree<ParticleType, typename ParticleType::PivotComp> kdtree_;

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

}  // namespace epf

#endif