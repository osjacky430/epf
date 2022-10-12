#ifndef DEFAULT_SAMPLER_HPP_
#define DEFAULT_SAMPLER_HPP_

#include "core/enum.hpp"
#include "core/measurement.hpp"
#include "core/process.hpp"
#include "core/state.hpp"
#include <type_traits>
#include <vector>

namespace epf {

/**
 *  @brief  Default importance sampler uses proposal distribution = p(x_{t}|x_{t-1},u_{t}) to calculate weight
 *          recursively. Which is the simplest way of doing it.
 */
template <typename State>
struct DefaultSampler {
  using StateVector = typename StateTraits<State>::ArithmeticType;

  template <typename T>
  static inline constexpr bool require_meas_model = std::is_base_of_v<MeasurementModel<State>, T>;

  SamplingResult importance_sampling(ProcessModel<State>* const t_process, MeasurementModel<State>* const t_meas,
                                     std::vector<StateVector>& t_prev, std::vector<double>& t_weight) {
    // If the state of interest didn't change, resampling should be suspended (and in fact it is usually a good idea
    // to suspend the integration of measurements as well). If we clearly know that the state remains the same, then
    // the diversity of the particles should also stay the same (what is unknown should remain unknown), resampling
    // process induces the loss of diversity, even though the variance of particle set decreases, the variance of the
    // particle set as an estimator of TRUE belief increases.
    if (t_process->predict(t_prev) == Prediction::NoUpdate) {
      return SamplingResult::NoSampling;
    }

    if (t_meas->update(t_prev, t_weight) == MeasurementResult::NoMeasurement) {
      return SamplingResult::NoSampling;
    }

    return SamplingResult::Sampled;
  }
};

}  // namespace epf

#endif