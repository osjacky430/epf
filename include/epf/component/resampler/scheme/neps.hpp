#ifndef NEPS_HPP_
#define NEPS_HPP_

#include <cstddef>
#include <range/v3/numeric/accumulate.hpp>
#include <vector>

namespace epf {

class NEPScheme {
  std::size_t resampling_threshold_ = 0;

 public:
  void set_resampling_threshold(std::size_t const t_thresh) noexcept { this->resampling_threshold_ = t_thresh; }

  [[nodiscard]] bool need_resample(std::vector<double>& t_weights) const noexcept {
    if (resampling_threshold_ == 0) {  // 0 indicate always resample
      return true;
    }

    auto const square_sum      = [](double const t_sum, double const t_weight) { return t_sum + t_weight * t_weight; };
    double const weight_sq_sum = ranges::accumulate(t_weights, 0.0, square_sum);

    return 1.0 / weight_sq_sum < static_cast<double>(this->resampling_threshold_);
  }
};

}  // namespace epf

#endif