#ifndef PARTICLE_HPP_
#define PARTICLE_HPP_

#include <cstddef>

namespace particle_filter::traits {

template <typename T>
struct coordinate_type {};

template <typename T>
struct coordinate_system {};

template <std::size_t Size>
struct dimension {
  static constexpr auto value = Size;
};

}  // namespace particle_filter::traits

#endif