#ifndef MAP_HPP_
#define MAP_HPP_

#include <array>
#include <utility>
#include <vector>

namespace epf {

template <typename ParticleType>
struct MapBase {
  MapBase() = default;

  MapBase(MapBase const&)     = default;
  MapBase(MapBase&&) noexcept = default;

  MapBase& operator=(MapBase const&) = default;
  MapBase& operator=(MapBase&&) noexcept = default;

  virtual ~MapBase() = default;

  virtual double distance_to_obstacle(typename ParticleType::ValueType const& /* t_pose */, double /* t_max_range */,
                                      std::array<double, 3> /* t_direciton */) const noexcept {
    // bresenham line algorithm
    return 0.0;
  }

  [[nodiscard]] virtual std::vector<std::array<double, 3>> get_free_cells() const noexcept { return {}; }

  [[nodiscard]] virtual std::vector<std::array<double, 3>> get_landmark() const noexcept { return {}; }

  [[nodiscard]] virtual bool is_free_cell(ParticleType const& /**/) const noexcept { return true; }
};

}  // namespace epf

#endif