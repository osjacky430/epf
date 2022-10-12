#ifndef AMCL_HPP_
#define AMCL_HPP_

#include "component/resampler/adaptive.hpp"
#include "component/sample_size_strategy/adaptive.hpp"
#include "core/particle_filter.hpp"
#include "util/traits.hpp"

namespace epf::amcl {

// TODO: separate pose to translation and rotation, if possible
struct Particle {
  using ValueType = std::array<double, 3>;  // x, y, w

  ValueType value_{};

  [[nodiscard]] bool operator==(Particle const& t_cmp) const noexcept { return this->value_ == t_cmp.value_; }

  [[nodiscard]] double operator[](std::size_t const t_idx) const noexcept { return this->value_[t_idx]; }
};

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

}  // namespace epf::amcl

namespace epf {

template <>
struct dimension_t<amcl::Particle> {
  static constexpr std::size_t value = 3;
};

}  // namespace epf

#endif