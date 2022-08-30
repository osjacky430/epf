#ifndef UKF_SAMPLER_HPP_
#define UKF_SAMPLER_HPP_

#include "core/measurement.hpp"
#include "core/process.hpp"

namespace epf {

template <typename State>
struct UKFSampler {
  MeasurementResult importance_sampling(ProcessModel<State>* const /**/, MeasurementModel<State>* const t_meas,
                                        std::vector<State>& t_prev, std::vector<double>& t_weight) {
    return t_meas->estimate(t_prev, t_weight);
  }
};

}  // namespace epf

#endif