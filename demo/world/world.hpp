#ifndef ZA_WARUDO_HPP_
#define ZA_WARUDO_HPP_

#include "map.hpp"

#include <filesystem>
#include <vector>

namespace demo {

struct World : epf::MapBase {
  double width_;
  double length_;
  double resolution_;

  static World from_image(std::filesystem::path const t_path) {
    //
  }
};

}  // namespace demo

#endif