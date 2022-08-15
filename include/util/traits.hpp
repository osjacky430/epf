#ifndef TRAITS_HPP_
#define TRAITS_HPP_

#include <cstddef>
#include <type_traits>

namespace epf {

template <typename T, typename I = std::size_t, typename = void>
struct has_subscript_operator : std::false_type {};

template <typename T, typename I>
struct has_subscript_operator<T, I, std::void_t<decltype(std::declval<T>()[std::declval<I>()])>> : std::true_type {};

template <typename T>
constexpr std::size_t dimension(T&& t_obj) noexcept {
  using std::size;
  return size(t_obj);
}

}  // namespace epf

#endif