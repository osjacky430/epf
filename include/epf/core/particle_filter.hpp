#ifndef PARTICLE_FILTER_HPP_
#define PARTICLE_FILTER_HPP_

#include "epf/component/resampler/resampler.hpp"
#include "epf/core/enum.hpp"
#include "epf/core/measurement.hpp"
#include "epf/core/process.hpp"
#include "epf/core/state.hpp"
#include "epf/util/math.hpp"
#include "epf/util/traits.hpp"
#include <memory>
#include <range/v3/algorithm/fill.hpp>
#include <range/v3/algorithm/generate.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/zip.hpp>
#include <type_traits>
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
template <typename State, typename ImportanceSampleStrategy, typename ResampleStrategy, typename MCMCStrategy = void>
class ParticleFilter : public ImportanceSampleStrategy, public ResampleStrategy {
  static_assert(is_specialization_of<Resampler, ResampleStrategy>);
  BOOST_CONCEPT_ASSERT((BasicStateConcept<State>));

 protected:
  using StateTrait  = StateTraits<State>;
  using StateVector = typename StateTrait::ArithmeticType;

  using ImportanceSampleStrategy::importance_sampling;
  using ResampleStrategy::resample;

  [[nodiscard]] auto get_all_measurement() const noexcept { return this->sensors_; }  // const& ?
  [[nodiscard]] auto get_process_model() const noexcept { return this->motion_.get(); }

 private:
  bool filter_updated_ = false;

  std::unique_ptr<ProcessModel<State>> motion_{};
  std::vector<std::shared_ptr<MeasurementModel<State>>> sensors_{};  // TODO: change to unique_ptr

  std::vector<typename StateTrait::ArithmeticType> previous_states_{};
  std::vector<double> state_weights_{};

 public:
  static constexpr auto DEFAULT_PARTICLE_COUNTS = 2000;

  explicit ParticleFilter(State t_initial_state, std::size_t const t_particle_counts = DEFAULT_PARTICLE_COUNTS)
    : previous_states_(t_particle_counts, to_state_vector(t_initial_state)),
      state_weights_(t_particle_counts, 1.0 / static_cast<double>(t_particle_counts)) {}

  explicit ParticleFilter(std::size_t const t_particle_counts)
    : previous_states_(t_particle_counts),
      state_weights_(t_particle_counts, 1.0 / static_cast<double>(t_particle_counts)) {}

  ParticleFilter(ParticleFilter const&)     = default;
  ParticleFilter(ParticleFilter&&) noexcept = default;

  ParticleFilter& operator=(ParticleFilter const&)     = default;
  ParticleFilter& operator=(ParticleFilter&&) noexcept = default;

  virtual ~ParticleFilter() = default;

  virtual State sample() noexcept {
    this->filter_updated_     = false;
    auto* const process_model = this->get_process_model();
    for (auto& measurement : this->get_all_measurement()) {
      // if either one of these isn't ready, then we won't get meaningful sampling / resampling
      if (not measurement->data_ready() or not process_model->input_ready()) {
        continue;
      }

      if (this->importance_sampling(process_model, measurement.get(), this->previous_states_, this->state_weights_) ==
          SamplingResult::NoSampling) {
        continue;
      }

      // we assumed that weight is already normalized here, @todo need to think
      // whether we should make this assumption or not
      this->filter_updated_ = true;  // make it true here regardless of resample stage because it might not take place
      this->resample(measurement.get(), this->previous_states_, this->state_weights_);
    }

    StateVector res = StateVector::Zero();
    res = ranges::accumulate(ranges::views::zip(this->previous_states_, this->state_weights_), res, weighted_sum);
    return from_state_vector<State>(res);
  }

  template <typename T>
  void set_initial_cond(T const& t_generator) noexcept {
    ranges::generate(this->previous_states_, t_generator);
    ranges::fill(this->state_weights_, 1.0 / static_cast<double>(this->previous_states_.size()));
  }

  [[nodiscard]] bool filter_updated() const noexcept { return this->filter_updated_; }

  template <typename Measurement, typename... InitArg>
  Measurement* add_measurement_model(InitArg&&... args) noexcept {
    static_assert(ImportanceSampleStrategy::template require_meas_model<Measurement>);
    this->sensors_.push_back(std::make_shared<Measurement>(std::forward<InitArg>(args)...));

    return static_cast<Measurement*>(this->sensors_.back().get());
  }

  template <typename Motion, typename... InitArg>
  Motion* set_process_model(InitArg&&... args) noexcept {
    static_assert(std::is_base_of_v<ProcessModel<State>, Motion>);
    this->motion_ = std::make_unique<Motion>(std::forward<InitArg>(args)...);

    return static_cast<Motion*>(this->motion_.get());
  }
};

}  // namespace epf

#endif