#if HAVE_MATPLOTCPP
#include <matplot/matplot.h>
#endif

#include "epf/app/common.hpp"
#include "epf/core/measurement.hpp"
#include "epf/core/process.hpp"
#include "pf_sim_cfg.hpp"

int main(int /**/, char** /**/) {
  using namespace time_series;

  Simulator<F> simulator{F::State{1.0}, TIME_STEP, 1};
  auto epf_tup = simulator.create_particle_filter<epf::ExtendedPF<F::State, F::Output>>(PARTICLE_COUNT);
}