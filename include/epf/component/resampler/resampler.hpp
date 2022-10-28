#ifndef RESAMPLER_HPP_
#define RESAMPLER_HPP_

#include "epf/core/measurement.hpp"
#include "epf/core/state.hpp"
#include <range/v3/numeric/accumulate.hpp>

namespace epf {

/**
 *  @brief
 *
 *  @note Policy class needs protected default destructor to prevent user delete ParticleFilter via pointer to this
 *        class
 */
template <typename State, typename Algo, typename Scheme>
class Resampler : public Algo, public Scheme {  // NOLINT(*-special-member-functions)
  using Algo::resample_impl;
  using Scheme::need_resample;

  bool filter_resampled_ = false;

 protected:
  ~Resampler() = default;

 public:
  using Algo::Algo;
  using StateVector = typename StateTraits<State>::ArithmeticType;

  void resample(epf::MeasurementModel<State>* const t_meas, std::vector<StateVector>& t_particles,
                std::vector<double>& t_weights) {
    this->filter_resampled_ = this->need_resample(t_weights);
    if (this->filter_resampled_) {
      this->resample_impl(t_meas, t_particles, t_weights);
    }
  }

  [[nodiscard]] bool filter_resampled() const noexcept { return this->filter_resampled_; }
};

}  // namespace epf

#endif