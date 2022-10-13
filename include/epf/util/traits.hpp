#ifndef TRAITS_HPP_
#define TRAITS_HPP_

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace epf {

template <typename T, typename I = std::size_t, typename = void>
struct has_subscript_operator : std::false_type {};

template <typename T, typename I>
struct has_subscript_operator<T, I, std::void_t<decltype(std::declval<T>()[std::declval<I>()])>> : std::true_type {};

template <typename T, typename I = std::size_t>
inline constexpr bool has_subscript_operator_v = has_subscript_operator<T, I>::value;

/**
 *  @brief  This function returns the dimension (number of state) of the object, e.g. for 2D localization, state to
 *          estimate can be [x, y, w], which means that dimension = 3, for custom type, create your own size() function
 *          that takes the object as parameter, and std::size_t as return value.
 *
 *  @code{.cpp}
 *
 *  namespace foo {
 *    struct bar {};  // particle type
 *
 *    constexpr std::size_t size(bar&& t_state) noexcept { return 3; }  // ADL
 *  }
 *
 *  @endcode
 */
template <typename T>
constexpr std::size_t dimension(T&& t_obj) noexcept {
  using std::size;
  return size(t_obj);
}

template <typename T>
struct dimension_t;

template <typename T>
inline constexpr std::size_t dimension_v = dimension_t<T>::value;

}  // namespace epf

#endif