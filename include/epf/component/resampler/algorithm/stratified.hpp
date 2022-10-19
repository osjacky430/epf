#ifndef STRATIFIED_RESAMPLE_HPP_
#define STRATIFIED_RESAMPLE_HPP_

#include "epf/component/resampler/resampler.hpp"
#include "epf/component/resampler/scheme/always.hpp"
#include "epf/core/measurement.hpp"
#include "epf/core/state.hpp"
#include "epf/util/math.hpp"
#include <random>
#include <range/v3/algorithm/lower_bound.hpp>

namespace epf {

template <typename State>
class Stratified {
  using StateVector = typename StateTraits<State>::ArithmeticType;
  std::mt19937 rng_ = std::mt19937(std::random_device{}());

  std::size_t resample_num_ = 0;

 public:
  void set_resample_size(std::size_t t_n) noexcept { this->resample_num_ = t_n; }

  // From Kitagawa 1996, Appendix: Resampling Algorithms
  //
  // @note I don't see Step 1 mentioned in UNSCENTED PARTICLE FILTER, nor is it implemented in particle_filter_tutorial
  //
  void resample_impl(epf::MeasurementModel<State>* const /**/, std::vector<StateVector>& t_previous_particles,
                     std::vector<double>& t_weight) noexcept {
    std::size_t const sample_count = this->resample_num_ != 0 ? this->resample_num_ : t_previous_particles.size();
    double const step_size         = 1.0 / static_cast<double>(sample_count);
    auto const cumulative_weight   = calculate_cumulative_weight(t_weight);
    if (cumulative_weight.back() == 0.0) {
      return;  // @todo this part needs to think thoroughly, all resampler
    }

    std::vector<StateVector> resampled_particle;
    resampled_particle.reserve(sample_count);
    for (auto iter = cumulative_weight.begin(); resampled_particle.size() < sample_count;) {
      std::uniform_real_distribution<> unfirom_dist(0, step_size);
      auto const rand_num = unfirom_dist(this->rng_) + step_size * resampled_particle.size();

      iter = ranges::lower_bound(iter, cumulative_weight.end(), rand_num);  // *iter >= rand_num
      resampled_particle.push_back(t_previous_particles[iter - cumulative_weight.begin()]);
    }

    t_previous_particles = std::move(resampled_particle);
    t_weight             = std::vector(sample_count, step_size);
  }
};

template <typename State, typename Scheme = AlwaysScheme>
using StratifiedResampler = Resampler<State, Stratified<State>, Scheme>;

}  // namespace epf

#endif