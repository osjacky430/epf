#ifndef AMCL_HPP_
#define AMCL_HPP_

#include "epf/component/resampler/adaptive.hpp"
#include "epf/component/sample_size_strategy/adaptive.hpp"
#include "epf/core/particle_filter.hpp"
#include "epf/util/traits.hpp"
#include <Eigen/Dense>
#include <cstddef>
#include <type_traits>

namespace epf {

struct Particle {
  struct TagType {};

  using ValueType = double;
  using Dimension = std::integral_constant<std::size_t, 3>;

  [[nodiscard]] bool operator==(Particle const& t_cmp) const noexcept { return this->pose_ == t_cmp.pose_; }

  std::array<ValueType, 3> pose_;
};

template <>
inline Particle from_state_vector(Eigen::Vector3d const& t_state_vec) {
  return Particle{std::array{t_state_vec[0], t_state_vec[1], t_state_vec[2]}};
}

template <>
inline Eigen::Vector3d to_state_vector(Particle const& t_state) {
  return Eigen::Vector3d{t_state.pose_[0], t_state.pose_[1], t_state.pose_[2]};
}

template <>
inline double& x_coor<Particle>(Eigen::Vector3d& t_arr) {
  return t_arr[0];
}

template <>
inline double& y_coor<Particle>(Eigen::Vector3d& t_arr) {
  return t_arr[1];
}

template <>
inline double& w_coor<Particle>(Eigen::Vector3d& t_arr) {
  return t_arr[2];
}

// [[nodiscard]] Pose transform_coordinate(Pose const& t_relative) const noexcept {
//     auto const& relative_pose = t_relative.pose_;
//     auto const translation    = Eigen::Vector2d(relative_pose[0], relative_pose[1]);
//     auto const rotation       = Eigen::Rotation2Dd(relative_pose[2]);
//     auto const origin         = Eigen::Array2d{this->pose_[0], this->pose_[1]};
//     auto transformed          = origin + (rotation * translation).array();
//     auto const angle          = this->pose_[2] + relative_pose[2];
//     return Pose{Eigen::Array3d{transformed[0], transformed[1], std::atan2(std::sin(angle), std::cos(angle))}};
//  }

template <typename State = Particle, typename ImportanceSampler = DefaultSampler<State>,
          typename Resampler = AdaptiveResample<State, KLDSampling>>
using AMCL2D = epf::ParticleFilter<State, ImportanceSampler, Resampler>;

}  // namespace epf

#endif