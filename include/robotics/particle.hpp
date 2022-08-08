#ifndef PARTICLE_HPP_
#define PARTICLE_HPP_

#include <cstddef>

namespace epf::traits {

template <typename T>
struct particle_type {};

template <typename T>
struct coordinate_type {};

template <typename T>
struct coordinate_system {};

template <std::size_t Size>
struct dimension {
  static constexpr auto value = Size;
};

template <typename T, std::size_t Idx, std::size_t Dimension>
struct access {};

}  // namespace epf::traits

namespace epf {

template <typename T>
struct ParticleAdapter_r {};

namespace traits {

template <typename T>
struct coordinate_type<ParticleAdapter_r<T>> {
  using type = double;
};

// template <typename T>
// struct dimension<> {};

}  // namespace traits

}  // namespace epf

#endif