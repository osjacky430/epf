#ifndef DEFAULT_RESAMPLER_HPP_
#define DEFAULT_RESAMPLER_HPP_

#include "core/measurement.hpp"
#include <algorithm>
#include <random>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/view/zip.hpp>
#include <vector>

namespace epf {

// only importance resampling
template <typename State>
struct DefaultResample {
  std::mt19937 rng_ = std::mt19937(std::random_device{}());

  void resample(epf::MeasurementModel<State>* const /**/, std::vector<State>& t_previous_particles,
                std::vector<double>& t_weight) noexcept {
    auto const cumulative_weight = [&]() {
      std::vector<double> ret_val = t_weight;
      for (std::size_t i = 1; i < t_weight.size(); ++i) {
        ret_val[i] += ret_val[i - 1];
      }

      return ret_val;
    }();

    std::vector<State> estimated_state(t_previous_particles.size());
    std::vector<double> state_weight(t_previous_particles.size());
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

    std::for_each(state_weight.begin(), state_weight.end(), [=](auto& t_val) { t_val /= total_weight; });

    t_previous_particles = std::move(estimated_state);
    t_weight             = std::move(state_weight);
  }
};

}  // namespace epf

#endif