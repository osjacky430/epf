#ifndef RESIDUAL_RESAMPLER_HPP_
#define RESIDUAL_RESAMPLER_HPP_

#include "epf/component/resampler/algorithm/multinomial.hpp"
#include "epf/component/resampler/scheme/always.hpp"
#include "epf/core/measurement.hpp"
#include "epf/core/state.hpp"
#include <cmath>
#include <iterator>
#include <range/v3/algorithm/fill_n.hpp>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/iterator/insert_iterators.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/zip.hpp>
#include <vector>

namespace epf {

template <typename State>
class Residual {
  using StateVector = typename StateTraits<State>::ArithmeticType;

  std::size_t resample_num_ = 0; /*!< The minimum amount of particle left after resampling, value 0 means that particle
                                    count remains the same after resampling */

 public:
  void set_resample_size(std::size_t t_n) noexcept { this->resample_num_ = t_n; }

  void resample_impl(epf::MeasurementModel<State>* const /**/, std::vector<StateVector>& t_previous_particles,
                     std::vector<double>& t_weight) noexcept {
    std::size_t const N = this->resample_num_ != 0 ? this->resample_num_ : t_previous_particles.size();

    std::vector<double> child_count = t_weight;
    ranges::for_each(child_count, [N](auto& t_weight) { t_weight = std::floor(t_weight * N); });

    // we can move particles with zero child count to the end of the vector, then overwrite them
    std::vector<StateVector> resampled_state;
    resampled_state.reserve(N);
    for (auto [state, child_count] : ranges::views::zip(t_previous_particles, child_count)) {
      ranges::fill_n(ranges::back_inserter(resampled_state), child_count, state);
    }

    if (Multinomial<State> sir; N > resampled_state.size()) {
      auto const Nt_comp = N - resampled_state.size();

      std::vector<double> new_weights = t_weight;
      for (auto [new_weight, child_count] : ranges::views::zip(new_weights, child_count)) {
        new_weight = (new_weight * N - child_count) / static_cast<double>(Nt_comp);
      }

      sir.set_resample_size(Nt_comp);
      sir.resample_impl(nullptr, t_previous_particles, new_weights);
      resampled_state.insert(resampled_state.end(), std::make_move_iterator(t_previous_particles.begin()),
                             std::make_move_iterator(t_previous_particles.end()));
    }

    t_previous_particles = std::move(resampled_state);
    t_weight             = std::vector<double>(N, 1.0 / N);  // @todo can we do better?
  }
};

template <typename State, typename Scheme = AlwaysScheme>
using ResidualResampler = Resampler<State, Residual<State>, Scheme>;

}  // namespace epf

#endif