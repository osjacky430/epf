#ifndef MOTION_MODEL_HPP_
#define MOTION_MODEL_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <random>

namespace particle_filter {

enum class UpdateStatus { Updated, NoUpdate };

template <typename ParticleType>
struct MotionModel {
  virtual UpdateStatus update_motion(std::vector<ParticleType>& t_current_pose) = 0;

  MotionModel()                   = default;
  MotionModel(MotionModel const&) = default;
  MotionModel& operator=(MotionModel const&) = default;

  MotionModel(MotionModel&&) noexcept = default;
  MotionModel& operator=(MotionModel&&) noexcept = default;

  virtual ~MotionModel() = default;
};

}  // namespace particle_filter

#endif