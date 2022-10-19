#ifndef ALWAYS_HPP_
#define ALWAYS_HPP_

#include <vector>

namespace epf {

struct AlwaysScheme {
  [[nodiscard]] static bool need_resample(std::vector<double>& /**/) noexcept { return true; }
};

}  // namespace epf

#endif