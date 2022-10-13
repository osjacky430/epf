#ifndef MULTINOMIAL_RESAMPLER_HPP_
#define MULTINOMIAL_RESAMPLER_HPP_

#include "epf/core/measurement.hpp"
#include "epf/core/state.hpp"
#include "epf/util/math.hpp"
#include <algorithm>
#include <random>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/view/zip.hpp>
#include <vector>

namespace epf {

/**
 *  @brief  This class implement default resampling strategy, i.e. Sampling-importance resampling (SIR), which is
 *          equivalent to sampling from multinomial distribution
 *
 *  Gordon 1994
 */
template <typename State>
class MultinomialResample {
  using StateVector = typename StateTraits<State>::ArithmeticType;
  std::mt19937 rng_ = std::mt19937(std::random_device{}());

  std::size_t resample_num_ = 0;

 public:
  void set_resample_size(std::size_t t_n) noexcept { this->resample_num_ = t_n; }

  void resample(epf::MeasurementModel<State>* const /**/, std::vector<StateVector>& t_previous_particles,
                std::vector<double>& t_weight) noexcept {
    std::size_t const sample_count = this->resample_num_ != 0 ? this->resample_num_ : t_previous_particles.size();

    auto const cumulative_weight = calculate_cumulative_weight(t_weight);
    std::vector<StateVector> estimated_state(sample_count);
    std::vector<double> state_weight(sample_count);
    if (cumulative_weight.back() == 0.0) {
      return;
    }

    double total_weight = 0.0;
    std::uniform_real_distribution<> uniform_dist(0.0, 1.0);

    ranges::for_each(ranges::views::zip(estimated_state, state_weight), [&](auto pair) {
      auto const iter = std::lower_bound(cumulative_weight.begin(), cumulative_weight.end(), uniform_dist(this->rng_));
      auto const chosen_idx = iter - cumulative_weight.begin();
      total_weight += t_weight[chosen_idx];
      std::get<0>(pair) = t_previous_particles[chosen_idx];
      std::get<1>(pair) = t_weight[chosen_idx];
    });

    ranges::for_each(state_weight, [=](auto& t_val) { t_val /= total_weight; });

    t_previous_particles = std::move(estimated_state);
    t_weight             = std::move(state_weight);
  }
};

}  // namespace epf

#endif