#ifndef PARTICLE_FILTER_HPP_
#define PARTICLE_FILTER_HPP_

#include "component/importance_sampler/default.hpp"
#include "component/resampler/default.hpp"
#include "core/enum.hpp"
#include "measurement.hpp"
#include "process.hpp"
#include <memory>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/zip.hpp>
#include <vector>

namespace epf {

/**
 *  @brief  This class is the interface of the Particle Filter, with SISR filter (sequential importance sampling with
 *          resampling, a.k.a BF, bootstrap filter, see [1]) as its default implementation. Inherit from this class if
 *          you want to use different type of filtering, such as Auxiliary Particle Filter (APF, see [2]).
 *
 *  [1] Gordon N, Salmond D, Smith AFM. 1993. Novel approach to nonlinear/non-Gaussian Bayesian state estimation. IEE
 *      Proceedings F. Radar Signal Process 140: 107–113.
 *  [2] Pitt MK, Shephard N. 1999. Filtering via simulation: auxiliary particle filters. Journal of the American
 *      Statistical Association 94: 590–599.
 */
template <typename State, typename ImportanceSampleStrategy = DefaultSampler<State>,
          typename ResampleStrategy = DefaultResample<State>>
class ParticleFilter : ImportanceSampleStrategy, ResampleStrategy {
 private:
  std::size_t sampled_time_ = 0; /*!< particle filter has sampled for this amount of time, this will get reset everytime
                                      resampling takes place */
  std::size_t resample_frequency_ = 1; /*!< do resample after sampling for this amount of time, resample_frequency_ <= 1
                                          means to force resample everytime sampling takes place */

 protected:
  using ImportanceSampleStrategy::importance_sampling;
  using ResampleStrategy::resample;

  auto get_all_measurement() const noexcept { return this->sensors_; }
  auto get_process_model() const noexcept { return this->motion_.get(); }

  std::vector<State> previous_states_{};
  std::vector<double> state_weights_{};

 public:
  static constexpr auto DEFAULT_MIN_PARTICLE_COUNTS = 100;
  static constexpr auto DEFAULT_MAX_PARTICLE_COUNTS = 2000;

  void set_resample_frequency(std::size_t const t_freq) noexcept { this->resample_frequency_ = t_freq; }

  // template <typename Distribution>
  // static auto init_particles(Distribution t_distribution, std::size_t const t_size) noexcept {
  //   std::vector<State> ret_val(t_size);
  //   std::generate(ret_val.begin(), ret_val.end(), [&]() { return State::random_particle(t_distribution); });

  //   return ret_val;
  // }

  explicit ParticleFilter(State t_initial_state, std::size_t const t_max_particle = DEFAULT_MAX_PARTICLE_COUNTS)
    : previous_states_(t_max_particle, t_initial_state),
      state_weights_(t_max_particle, 1.0 / static_cast<double>(t_max_particle)) {}

  ParticleFilter(ParticleFilter const&)     = default;
  ParticleFilter(ParticleFilter&&) noexcept = default;

  ParticleFilter& operator=(ParticleFilter const&) = default;
  ParticleFilter& operator=(ParticleFilter&&) noexcept = default;

  virtual ~ParticleFilter() = default;

  virtual State sample() noexcept {
    for (auto& measurement : this->get_all_measurement()) {
      auto process_model = this->get_process_model();
      if (this->importance_sampling(process_model, measurement.get(), this->previous_states_, this->state_weights_) ==
          SamplingResult::NoSampling) {
        continue;
      }

      if (++this->sampled_time_ >= this->resample_frequency_) {
        this->sampled_time_ = 0;
        this->resample(measurement.get(), this->previous_states_, this->state_weights_);
      }
    }

    auto const weighted_sum = [](auto const& t_left, auto const& t_right) {
      return t_left + t_right.first * t_right.second;
    };

    return ranges::accumulate(ranges::views::zip(this->previous_states_, this->state_weights_), State{}, weighted_sum);
  }

  template <typename Measurement, typename... InitArg>
  void add_measurement_model(InitArg&&... args) noexcept {
    this->sensors_.push_back(std::make_shared<Measurement>(std::forward<InitArg>(args)...));
  }

  template <typename Motion, typename... InitArg>
  void set_process_model(InitArg&&... args) noexcept {
    this->motion_ = std::make_unique<Motion>(std::forward<InitArg>(args)...);
  }

 private:
  std::unique_ptr<ProcessModel<State>> motion_{};
  std::vector<std::shared_ptr<MeasurementModel<State>>> sensors_{};  // TODO: change to unique_ptr
};

}  // namespace epf

#endif