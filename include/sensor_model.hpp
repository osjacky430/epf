#ifndef SENSOR_MODEL_HPP_
#define SENSOR_MODEL_HPP_

#include "map.hpp"

#include <array>
#include <vector>

namespace particle_filter {

enum class SensorResult { NoMeasurement, Estimated };

/**
 *  @brief  Abstract interface for sensor model that operate on ParticleType
 *
 */
template <typename ParticleType>
struct SensorModel {
  // MapBase makes particle filter usable only in robotics
  virtual SensorResult estimate(std::vector<ParticleType>& t_pose, MapBase<ParticleType>& t_map) = 0;

  virtual ParticleType sample_pose_from_latest_measurement(MapBase<ParticleType> const& t_map) = 0;

  SensorModel()                   = default;
  SensorModel(SensorModel const&) = default;
  SensorModel& operator=(SensorModel const&) = default;

  SensorModel(SensorModel&&) noexcept = default;
  SensorModel& operator=(SensorModel&&) noexcept = default;

  virtual ~SensorModel() = default;
};

}  // namespace particle_filter

#endif