#if HAVE_MATPLOTCPP
#include <matplot/matplot.h>
#endif

#include "epf/app/common.hpp"
#include "epf/core/measurement.hpp"
#include "epf/core/process.hpp"
#include "pf_sim_cfg.hpp"
#include <cmath>
#include <cstdlib>
#include <fmt/format.h>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/transform.hpp>

int main(int /**/, char** /**/) {
  using namespace time_series;

  Simulator<F> simulator{F::State{1.0}, TIME_STEP, 1};
  auto epf_tup = simulator.create_particle_filter<epf::ExtendedPF<F::State, F::Output>>(PARTICLE_COUNT);

  auto& epf = std::get<0>(epf_tup);

  auto* process = std::get<1>(epf_tup);
  process->set_process_covariance(Eigen::Matrix<double, 1, 1>{1});
  process->set_noise_covariance(Eigen::Matrix<double, 1, 1>{{F::STATE_NOISE_COV}});

  auto* meas = std::get<2>(epf_tup);
  meas->set_noise_covariance(Eigen::Matrix<double, 1, 1>{F::OBSERVATION_NOISE});

  std::vector<double> estimated_state(1, 1.0);
  simulator.simulate([&](std::size_t const /**/) { return epf.sample(); },
                     [&](auto const& t_v) { estimated_state.push_back(t_v.x_); });

  auto const time_series_state     = simulator.get_time_series();
  auto const generated_time_series = time_series_state | ranges::views::drop(1) |
                                     ranges::views::transform([](auto t_state) { return t_state.x_; }) |
                                     ranges::to_vector;

  double mse = 0.0;
  ranges::for_each(ranges::views::zip(estimated_state, generated_time_series), [&](auto t_zip) {
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