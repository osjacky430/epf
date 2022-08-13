#ifndef DEFAULT_RESAMPLER_HPP_
#define DEFAULT_RESAMPLER_HPP_

#include <algorithm>
#include <random>
#include <vector>

namespace epf {

// only importance resampling
template <typename ParticleType>
struct DefaultResample {
  std::mt19937 rng_ = std::mt19937(std::random_device{}());

  std::vector<ParticleType> resample(std::vector<ParticleType> const& t_previous_particles) const noexcept {
    auto const cumulative_weight = [&]() {
      std::vector<double> ret_val(t_previous_particles.size() + 1, 0);
      for (std::size_t i = 0; i < t_previous_particles.size(); ++i) {
        ret_val[i + 1] = t_previous_particles[i].weight() + ret_val[i];
      }

      return ret_val;
    }();

    std::vector<ParticleType> ret_val(t_previous_particles.size());
    if (cumulative_weight.back() == 0.0) {  // cumulative_weight.back() == weight sum
      return ret_val;
    }

    double total_weight = 0.0;
    std::uniform_real_distribution<> uniform_dist(0.0, 1.0);

    std::generate(ret_val.begin(), ret_val.end(), [&]() {
      auto const iter = std::lower_bound(cumulative_weight.begin(), cumulative_weight.end(), uniform_dist(this->rng_));
      auto chosen_particle = t_previous_particles[iter - cumulative_weight.begin()];
      total_weight += chosen_particle.weight();
      return chosen_particle;
    });

    std::for_each(ret_val.begin(), ret_val.end(), [=](auto& t_particle) { t_particle.weight() /= total_weight; });
    return ret_val;
  }
};

}  // namespace epf

#endif