#ifndef MEASUREMENT_HPP_
#define MEASUREMENT_HPP_

#include "enum.hpp"
#include <optional>
#include <vector>

namespace epf {

/**
 *  @brief
 */
template <typename T>
struct Measurement {
  virtual std::optional<T> get_measurement() = 0;

  Measurement()                   = default;
  Measurement(Measurement const&) = default;
  Measurement& operator=(Measurement const&) = default;

  Measurement(Measurement&&) noexcept = default;
  Measurement& operator=(Measurement&&) noexcept = default;

  virtual ~Measurement() = default;
};

/**
 *  @brief  Abstract interface for measurement model, formalized as z_{k} = h_{k}(x_{k}, u_{k}, n_{k})
 *
 */
template <typename State>
struct MeasurementModel {
  virtual MeasurementResult update(std::vector<State>& t_states, std::vector<double>& t_weight) = 0;

  /**
   *  @brief  This function generate random particle and assign weight to it according to latest measurement. This is
   *          used to add diversity in partcles so that particle filter can recover from incorrect state estimation. Not
   *          all algorithm require this to be implemented, as it is just one of the proposals to the problem mentioned
   *          above.
   *
   *  @todo   Is there a better way to opt in implementation?
   */
  virtual std::pair<State, double> sample_state_from_latest_measurement() { return {}; }

  MeasurementModel()                        = default;
  MeasurementModel(MeasurementModel const&) = default;
  MeasurementModel& operator=(MeasurementModel const&) = default;

  MeasurementModel(MeasurementModel&&) noexcept = default;
  MeasurementModel& operator=(MeasurementModel&&) noexcept = default;

  virtual ~MeasurementModel() = default;
};

}  // namespace epf

#endif