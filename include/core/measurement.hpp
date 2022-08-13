#ifndef MEASUREMENT_HPP_
#define MEASUREMENT_HPP_

#include "map.hpp"

#include <optional>

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

enum class MeasurementResult { NoMeasurement, Estimated };

/**
 *  @brief  Abstract interface for measurement model, formalized as z_{k} = h_{k}(x_{k}, u_{k}, n_{k})
 *
 */
template <typename State>
struct MeasurementModel {
  virtual MeasurementResult estimate(std::vector<State>& t_pose) = 0;

  virtual State sample_state_from_latest_measurement() = 0;

  MeasurementModel()                        = default;
  MeasurementModel(MeasurementModel const&) = default;
  MeasurementModel& operator=(MeasurementModel const&) = default;

  MeasurementModel(MeasurementModel&&) noexcept = default;
  MeasurementModel& operator=(MeasurementModel&&) noexcept = default;

  virtual ~MeasurementModel() = default;
};

}  // namespace epf

#endif