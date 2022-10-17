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

}  // namespace epf

#endif