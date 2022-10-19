#ifndef ADAPTIVE_SAMPLE_SIZE_HPP_
#define ADAPTIVE_SAMPLE_SIZE_HPP_

#include "epf/core/kdtree.hpp"
#include "epf/core/state.hpp"
#include "epf/util/traits.hpp"
#include <algorithm>
#include <boost/math/special_functions.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>

namespace epf {

struct DefaultTreeComp {
  template <typename State>
  bool operator()(State const& t_lhs, State const& t_rhs,
                  std::pair<double, std::size_t>* const t_pivot) const noexcept {
    constexpr auto STATE_DIM = StateTraits<State>::Dimension::value;
    if (t_pivot->second < STATE_DIM) {
      return t_pivot->first < t_rhs[t_pivot->second];
    }

    for (std::size_t idx = 0; idx < STATE_DIM; ++idx) {
      if (auto const abs_val = std::abs(t_lhs[idx] - t_rhs[idx]); abs_val > t_pivot->first) {
        t_pivot->first  = abs_val;
        t_pivot->second = idx;
      }
    }

    t_pivot->first = (t_lhs[t_pivot->second] + t_rhs[t_pivot->second]) / 2.0;  // mean split

    return t_pivot->first < t_rhs[t_pivot->second];
  }
};

template <typename State, typename TreeComp = DefaultTreeComp>
class KLDSampling {
  using StateVector = typename StateTraits<State>::ArithmeticType;

  static inline constexpr auto STATE_DIM       = StateTraits<State>::Dimension::value;
  static inline constexpr auto DEFAULT_EPSILON = 0.01;

  Eigen::Array<double, STATE_DIM, 1> bin_size_;

  /* !< Maximum allowed error (K-L distance) between true and estimated distribution */
  double epsilon_ = DEFAULT_EPSILON;

  // according to standard normal distribution wikipedia. A normal random variable X will exceed mu + z_{1-p} * sigma
  // with probability p (i.e. within mu + z_{1-p} with probability 1 - p), and will lie outside the interval [mu -
  // z_{1-p} * sigma, mu + z_{1-p} * sigma] with the probability 2 * p. (p is changed to 1 - p to match the notation
  // used in the paper), default upper quantile = 3 corresponds to p = 0.001, i.e. 1 - p = 0.999
  static inline constexpr auto DEFAULT_UPPER_QUANTILE = 3;

  /* !< z_{1-p}, upper quantile of standard normal distribution N(0, 1). where 1 - p is the probability that K-L
   *    distance between MLE and the true distribution is smaller than epsilon */
  double upper_quantile_ = DEFAULT_UPPER_QUANTILE;

  std::size_t min_particle_count_{};
  std::size_t max_particle_count_{};

  /* !< Data structure to store histogram */
  epf::KDTree<State, TreeComp> kdtree_;

 protected:
  ~KLDSampling() = default;  // prevent base pointer delete via policy class

  /**
   *  @brief  This function inserts particle, divided by resolution (a.k.a bin size in our
   *          application), to kdtree, then calculate particle size accordingly
   *
   *  @note   According to [1], the bin size is better fixed at 50 (cm) * 50 (cm) * 10 (deg) for 2D case
   *
   *  [1] Fox, Dieter. "Adapting the Sample Size in Particle Filters Through KLD-Sampling"
   */
  [[nodiscard]] std::size_t calculate_particle_size(std::vector<StateVector> const& t_particles) noexcept {
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

 public:
  [[nodiscard]] std::size_t max_particle() const noexcept { return this->max_particle_count_; }
  [[nodiscard]] std::size_t min_particle() const noexcept { return this->min_particle_count_; }

  void set_max_particle_count(std::size_t const t_max) noexcept { this->max_particle_count_ = t_max; }
  void set_min_particle_count(std::size_t const t_min) noexcept { this->min_particle_count_ = t_min; }
  void set_bin_size(Eigen::Array<double, STATE_DIM, 1> const& t_bin_size) noexcept { this->bin_size_ = t_bin_size; }
  void set_approximation_error(double const t_epsilon) noexcept { this->epsilon_ = t_epsilon; }

  void set_upper_quantile_from_prob(double const t_prob) noexcept {
    double const p        = t_prob > 0.5 ? t_prob : 1.0 - t_prob;
    this->upper_quantile_ = std::sqrt(2) * boost::math::erf_inv(2 * p - 1);
  }

  // cluster here is a bit difficult since we remove weight from State, we need to find other way to update weight if
  // we insert same element into tree
  void cluster_particles() {}

  KLDSampling() = default;
};

}  // namespace epf

#endif