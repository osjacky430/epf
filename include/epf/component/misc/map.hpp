#ifndef MAP_HPP_
#define MAP_HPP_

#include <Eigen/Dense>
#include <array>
#include <utility>
#include <vector>

namespace epf {

template <std::size_t>
struct Dimension;

template <>
struct Dimension<2> {
  using PositionType = Eigen::Vector2d;
  using PoseType     = Eigen::Vector3d;
  using VectorType   = Eigen::Vector2d;
  using DistanceType = double;
};

template <>
struct Dimension<3> {
  using PositionType = Eigen::Vector3d;
  using PoseType     = Eigen::Vector<double, 6>;
  using VectorType   = Eigen::Vector3d;
  using DistanceType = double;
};

template <typename Dim>
struct LocationMap {
  LocationMap() = default;

  LocationMap(LocationMap const&)     = default;
  LocationMap(LocationMap&&) noexcept = default;

  LocationMap& operator=(LocationMap const&)     = default;
  LocationMap& operator=(LocationMap&&) noexcept = default;

  virtual ~LocationMap() = default;

  using PoseType     = typename Dim::PoseType;
  using DistanceType = typename Dim::DistanceType;
  using VectorType   = typename Dim::VectorType;

  virtual double distance_to_obstacle(PoseType const& /* t_pose */, DistanceType /* t_max_range */,
                                      VectorType /* t_direciton */) = 0;

  virtual std::vector<PoseType> get_free_cells() = 0;

  virtual bool is_free_cell(PoseType const& /**/) const noexcept = 0;
};

template <typename Dim, typename Signature>
struct LandmarkMap {
  LandmarkMap() = default;

  LandmarkMap(LandmarkMap const&)     = default;
  LandmarkMap(LandmarkMap&&) noexcept = default;

  LandmarkMap& operator=(LandmarkMap const&)     = default;
  LandmarkMap& operator=(LandmarkMap&&) noexcept = default;

  virtual ~LandmarkMap() = default;

  using PoseType = typename Dim::PoseType;

  virtual PoseType get_landmark(Signature const& /**/) = 0;
};

template <typename Signature>
using LandmarkMap2D = LandmarkMap<Dimension<2>, Signature>;

using LocationMap2D = LocationMap<Dimension<2>>;

}  // namespace epf

#endif