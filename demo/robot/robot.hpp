#ifndef ROBOT_HPP_
#define ROBOT_HPP_

#include "map.hpp"
#include <array>
#include <cstddef>
#include <vector>

namespace demo {

struct LaserRanger2D {
  std::size_t sample_;
  double range_;
  double angle_;

  std::vector<double> generate_measurement(particle_filter::MapBase const& t_map, std::array<double, 3> const& t_pose) {
    //
  }
};

struct Robot {
  double x_;
  double y_;
  double theta_;

  double linear_motion_sigma_;
  double angular_motion_sigma_;

  std::array<LaserRanger2D, 2> lidar_;

  void move() {}

  void measurement() {}
};

}  // namespace demo

#endif