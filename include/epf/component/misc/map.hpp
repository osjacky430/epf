#ifndef MAP_HPP_
#define MAP_HPP_

#include <array>
#include <utility>
#include <vector>

namespace epf {

template <std::size_t>
struct Dimension;

template <>
struct Dimension<2> {
  using PositionType = std::array<double, 2>;
  using PoseType     = std::array<double, 3>;
  using VectorType   = std::array<double, 2>;
  using DistanceType = double;
};

template <>
struct Dimension<3> {
  using PositionType = std::array<double, 3>;
  using PoseType     = std::array<double, 6>;
  using VectorType   = std::array<double, 3>;
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

  virtual double distance_to_obstacle(typename Dim::PoseType const& /* t_pose */, double /* t_max_range */,
                                      typename Dim::VectorType /* t_direciton */) = 0;

  virtual std::vector<typename Dim::PoseType> get_free_cells() = 0;

  virtual bool is_free_cell(typename Dim::PoseType const& /**/) const noexcept = 0;
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

  virtual typename Dim::PoseType get_landmark(Signature const& /**/) = 0;
};

template <typename Signature>
using LandmarkMap2D = LandmarkMap<Dimension<2>, Signature>;

}  // namespace epf

#endif