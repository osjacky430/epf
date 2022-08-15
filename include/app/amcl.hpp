#ifndef AMCL_HPP_
#define AMCL_HPP_

#include "component/resampler/adaptive.hpp"
#include "component/sample_size_strategy/adaptive.hpp"
#include "core/particle_filter.hpp"

namespace epf::amcl {

struct KDTreeComp;

template <typename T, typename PC = KDTreeComp>
struct Particle final : public ParticleAdapter<T> {
  using ValueType = T;
  using PivotComp = PC;

  friend PivotComp;

  Particle() = default;
  using ParticleAdapter<T>::ParticleAdapter;

  void update(Particle const& t_pt) noexcept { this->weight_ += t_pt.weight(); }

  static constexpr auto PARTICLE_DIMENSION = dimension(std::declval<T>());

 private:
  std::size_t pivot_idx_ = PARTICLE_DIMENSION;
  double pivot_value_    = 0.0;
};

struct KDTreeComp {
  template <typename T>
  bool operator()(Particle<T, KDTreeComp>& t_lhs, Particle<T, KDTreeComp> const& t_rhs) const noexcept {
    // if (t_lhs.pivot_idx_ < traits::dimension<T>::size) {
    //   return t_lhs.pivot_value_ < t_rhs.value_.get(t_lhs.pivot_idx_);
    // }

    // double max_diff          = 0.0;
    std::size_t max_diff_idx = 0;
    // for (std::size_t idx = 0; idx < traits::dimension<T>::size; ++idx) {
    //   auto const diff = t_lhs.value_.get(idx) - t_rhs.value_.get(idx);
    //   if (std::abs(diff) > max_diff) {
    //     max_diff     = std::abs(diff);
    //     max_diff_idx = idx;
    //   }
    // }

    t_lhs.pivot_idx_   = max_diff_idx;
    t_lhs.pivot_value_ = (t_lhs.value_.get(max_diff_idx) + t_rhs.value_.get(max_diff_idx)) / 2.0;  // mean split

    return t_lhs.pivot_value_ < t_rhs.value_.get(max_diff_idx);
  }
};

template <typename ParticleType, typename Resampler = AdaptiveResample<ParticleType, KLDSampling>>
class AMCL2D final : public epf::ParticleFilter<ParticleType, Resampler> {
 private:
  std::vector<ParticleType> previous_particles_{};

  using PF = epf::ParticleFilter<ParticleType, Resampler>;

 public:
  explicit AMCL2D(std::size_t t_max_particles = PF::DEFAULT_MAX_PARTICLE_COUNTS)
    : previous_particles_(t_max_particles, ParticleType(1.0 / static_cast<double>(t_max_particles))) {}

  ParticleType sample() noexcept override {
    for (auto& measurement : this->get_all_measurement()) {
      // When the robot stopped, resampling should be suspended (and in fact it is usually a good idea to suspend the
      // integration of measurements as well). If we clearly know that the robot doesn't move at all, then the diversity
      // of the particles should remain the same (what remains unknown should still be unknown), resampling process
      // induces the loss of diversity, even though the variance of particle set decreases, the variance of the particle
      // set as an estimator of TRUE belief increases.
      if (auto status = this->get_process_model()->predict(this->previous_particles_);
          status == epf::Prediction::NoUpdate) {
        continue;
      }

      auto const estimate_result = measurement->estimate(this->previous_particles_);
      if (estimate_result == epf::MeasurementResult::NoMeasurement) {  // no valid estimation yet
        continue;
      }

      this->previous_particles_ = this->resample(measurement.get(), this->previous_particles_);
    }

    auto const weighted_sum = [](auto const& t_left, auto const& t_right) {
      return t_left.value() + t_right.weighted_value();
    };

    return std::accumulate(this->previous_particles_.begin(), this->previous_particles_.end(), ParticleType{},
                           weighted_sum);
  }
};

}  // namespace epf::amcl

#endif