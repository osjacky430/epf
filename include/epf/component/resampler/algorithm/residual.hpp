#ifndef RESIDUAL_RESAMPLER_HPP_
#define RESIDUAL_RESAMPLER_HPP_

#include "epf/component/resampler/algorithm/multinomial.hpp"
#include "epf/component/resampler/scheme/always.hpp"
#include "epf/core/measurement.hpp"
#include "epf/core/state.hpp"
#include <cmath>
#include <iterator>
#include <random>
#include <range/v3/algorithm/fill_n.hpp>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/iterator/insert_iterators.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#include <vector>

namespace epf {

template <typename State>
class Residual {
  std::size_t resample_num_ = 0; /*!< The minimum amount of particle left after resampling, value 0 means that particle
                                    count remains the same after resampling */

  Multinomial<State> sir_;  // this will become a problem in the future since we will make all policy class protected
                            // default destructor to prevent user accidentally delete particle filter via its base class

 public:
  Residual() = default;
  explicit Residual(std::mt19937 const& t_rng) : sir_{t_rng} {}

  using StateVector = typename StateTraits<State>::ArithmeticType;

  void set_resample_size(std::size_t t_n) noexcept { this->resample_num_ = t_n; }

  void resample_impl(epf::MeasurementModel<State>* const /**/, std::vector<StateVector>& t_previous_particles,
                     std::vector<double>& t_weight) noexcept {
    std::size_t const N = this->resample_num_ != 0 ? this->resample_num_ : t_previous_particles.size();

    std::vector<StateVector> resampled_state;
    resampled_state.reserve(N);
    for (auto [state, weight] : ranges::views::zip(t_previous_particles, t_weight)) {
      auto const child_count = static_cast<std::size_t>(std::floor(weight * N));

      ranges::fill_n(ranges::back_inserter(resampled_state), child_count, state);
      weight = weight * N - child_count;
    }

    // if N == resampled_state.size(), we must skip sir here, otherwise sir_.set_resample_size(0) will result in
    // sir_.resample_impl(...) do resampling base on size of sample set
    if (N > resampled_state.size()) {
      auto const Nt_comp = N - resampled_state.size();
      ranges::for_each(t_weight, [div = static_cast<double>(Nt_comp)](auto& t_val) { t_val /= div; });

      this->sir_.set_resample_size(Nt_comp);
      this->sir_.resample_impl(nullptr, t_previous_particles, t_weight);
      resampled_state.insert(resampled_state.end(), std::make_move_iterator(t_previous_particles.begin()),
                             std::make_move_iterator(t_previous_particles.end()));
    }

    t_previous_particles = std::move(resampled_state);       // @todo can we do better?
    t_weight             = std::vector<double>(N, 1.0 / N);  // @todo can we do better?
  }
};

template <typename State, typename Scheme = AlwaysScheme>
using ResidualResampler = Resampler<State, Residual<State>, Scheme>;

}  // namespace epf

#endif