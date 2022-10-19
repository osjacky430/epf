#ifndef RESAMPLER_HPP_
#define RESAMPLER_HPP_

#include "epf/core/measurement.hpp"
#include "epf/core/state.hpp"
#include <range/v3/numeric/accumulate.hpp>

namespace epf {

template <typename State, typename Algo, typename Scheme>
struct Resampler : public Algo, public Scheme {
  using StateVector = typename StateTraits<State>::ArithmeticType;

  using Algo::resample_impl;
  using Scheme::need_resample;

  void resample(epf::MeasurementModel<State>* const t_meas, std::vector<StateVector>& t_particles,
                std::vector<double>& t_weights) {
    if (this->need_resample(t_weights)) {
      this->resample_impl(t_meas, t_particles, t_weights);
    }
  }
};

}  // namespace epf

#endif