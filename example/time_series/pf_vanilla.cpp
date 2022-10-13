/**
 *
 *
 *
 *
 */

#if HAVE_MATPLOTCPP
#include <matplot/matplot.h>
#endif

#include "epf/core/particle_filter.hpp"
#include "pf_sim_cfg.hpp"
#include "time_series.hpp"
#include <algorithm>
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <cstdlib>
#include <fmt/format.h>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>

using ranges::for_each;
using ranges::to_vector;
using ranges::views::drop;
using ranges::views::transform;
using ranges::views::zip;

int main(int /**/, char** /**/) {
  using namespace time_series;

  Simulator<F> simulator{F::State{1.0}, TIME_STEP, 1};
  auto pf = std::get<0>(simulator.create_particle_filter<epf::ParticleFilter<F::State>>(PARTICLE_COUNT));

  double mse = 0.0;
  std::vector<double> estimated_state(1, 1.0);
  simulator.simulate([&](std::size_t /**/) mutable { return pf.sample().x_; },
                     [&](auto const& t_v) { estimated_state.push_back(t_v); });

  auto const time_series_state = simulator.get_time_series();
  auto const generated_time_series =
    time_series_state | drop(1) | transform([](auto const& t_state) { return t_state.x_; }) | to_vector;

  for_each(zip(estimated_state, generated_time_series), [&](auto t_zip) {
    auto const [estimated, current] = t_zip;
    mse += std::pow(current - estimated, 2);
    fmt::print("current value: {: f}, estimated value: {: f}, diff: {: f}\n", current, estimated, current - estimated);
  });

  mse /= TIME_STEP;
  fmt::print("MSE PF: {: .7}\n", mse);

#if HAVE_MATPLOTCPP
  auto const x_axis = matplot::linspace(1, TIME_STEP, TIME_STEP);
  matplot::plot(x_axis, generated_time_series, "o", x_axis, estimated_state, "g-");
  matplot::legend({"True x", "PF estimate"});
  matplot::xlabel("Time");
  matplot::ylabel("E[x(t)]");
  matplot::show();
#endif

  return EXIT_SUCCESS;
}