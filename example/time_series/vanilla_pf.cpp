/**
 *
 *
 *
 *
 */

#if HAVE_MATPLOTCPP
#include <matplot/matplot.h>
#endif

#include "core/measurement.hpp"
#include "core/particle_filter.hpp"
#include "core/process.hpp"
#include <algorithm>
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <fmt/format.h>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>

namespace time_series::vanilla_pf {

inline constexpr auto PARTICLE_COUNT = 200;
inline constexpr auto TIME_STEP      = 60;
inline constexpr auto PI             = boost::math::double_constants::pi;
inline constexpr double OMEGA        = 4e-2;
inline constexpr double PHI_1        = 0.5;
inline constexpr double PHI_2        = 0.2;
inline constexpr double PHI_3        = 0.5;

// c++ gamma distribution takes parameter alpha, beta but in fact, it means k and theta..
// From bayesian statistics point of view, k = alpha, and theta = 1.0 / beta..
inline constexpr double ALPHA = 3.0;
inline constexpr double BETA  = 0.5;

inline constexpr double OBSERVATION_NOISE = 1e-5;

// @TODO: create a simulation env helper?
class TimeStep {
  std::size_t current_;

 public:
  explicit TimeStep(std::size_t const t_start) : current_(t_start) {}

  void update_time_step() noexcept { ++this->current_; }

  template <typename T>
  [[nodiscard]] T current_time_step() const noexcept {
    return static_cast<T>(this->current_);
  }
};

// @todo come back later, perhaps encapsulate with simulation helper
static TimeStep& get_global_time() {
  static TimeStep GLOBAL_TIME{1};
  return GLOBAL_TIME;
}

template <typename T>
static T get_current_time_step() {
  return get_global_time().current_time_step<T>();
}

struct State {
  double x;

  State operator*(double const t_v) const noexcept { return State{this->x * t_v}; }
  State operator+(State const t_rhs) const noexcept { return State{this->x + t_rhs.x}; }
};

class StateTransition final : public epf::ProcessModel<State> {
  std::mt19937 rng_{std::random_device{}()};

 public:
  StateTransition() = default;

  [[nodiscard]] epf::Prediction predict(std::vector<State>& t_state) override {
    std::gamma_distribution<> dist(ALPHA, BETA);
    std::for_each(t_state.begin(), t_state.end(), [&](auto& t_val) mutable {
      t_val.x = 1 + std::sin(OMEGA * PI * get_current_time_step<double>()) + PHI_1 * t_val.x + dist(this->rng_);
    });

    return epf::Prediction::Updated;
  }
};

class Observer {
  std::mt19937 rng_{std::random_device{}()};

  std::vector<State> states_;
  std::vector<State> observed_;
  std::gamma_distribution<> dist{ALPHA, BETA};
  std::normal_distribution<> norm{0.0, std::sqrt(OBSERVATION_NOISE)};

  [[nodiscard]] State transition_function(State const t_prev, std::size_t t_time) noexcept {
    return State{1.0 + std::sin(OMEGA * PI * static_cast<double>(t_time)) + dist(this->rng_)} + t_prev * PHI_1;
  }

  [[nodiscard]] State observation_function(State const t_state, std::size_t t_time) noexcept {
    if (t_time > TIME_STEP / 2) {
      return t_state * PHI_3 + State{-2 + norm(this->rng_)};
    }

    return State{std::pow(t_state.x, 2) * PHI_2 + norm(this->rng_)};
  }

 public:
  Observer(std::size_t t_end, State t_initial) : states_(1, t_initial), observed_(1, t_initial) {
    for (std::size_t i = 1; i <= t_end; ++i) {
      this->states_.push_back(this->transition_function(this->states_.back(), i));
      this->observed_.push_back(this->observation_function(this->states_.back(), i + 1));
    }
  }

  [[nodiscard]] std::vector<State> get_time_series() const noexcept { return this->states_; }

  [[nodiscard]] State get_measurement(std::size_t const t_time) noexcept { return this->observed_.at(t_time); }
};

class Observation final : public epf::MeasurementModel<State> {
  Observer& obs_;

  std::mt19937 rng_{std::random_device{}()};

 public:
  explicit Observation(Observer& t_obs) : obs_{t_obs} {}

  [[nodiscard]] epf::MeasurementResult update(std::vector<State>& t_states, std::vector<double>& t_weight) override {
    auto const observation = this->obs_.get_measurement(get_current_time_step<std::size_t>());

    std::normal_distribution<> dist{0, std::sqrt(OBSERVATION_NOISE)};
    for (auto [state, weight] : ranges::views::zip(t_states, t_weight)) {
      auto const theoretical_observation = [this, rv = dist(this->rng_)](auto t_s) {
        if (get_current_time_step<std::size_t>() > TIME_STEP / 2) {
          return t_s.x * PHI_3 - 2 + rv;
        }

        return std::pow(t_s.x, 2) * PHI_2 + rv;
      }(state);

      weight = std::exp(-std::pow(observation.x - theoretical_observation, 2) / (2 * OBSERVATION_NOISE));
    }

    auto const weight_sum = std::accumulate(t_weight.begin(), t_weight.end(), 0.0);
    auto const normalize  = [weight_sum, particle_size = t_weight.size()](auto& t_val) {
      if (weight_sum == 0) {
        t_val = 1.0 / static_cast<double>(particle_size);
        return;
      }

      t_val /= weight_sum;
    };
    std::for_each(t_weight.begin(), t_weight.end(), normalize);

    return epf::MeasurementResult::Estimated;
  }
};

}  // namespace time_series::vanilla_pf

namespace vanilla_pf = time_series::vanilla_pf;

using ranges::for_each;
using ranges::to_vector;
using ranges::views::transform;
using ranges::views::zip;

int main(int /**/, char** /**/) {
  vanilla_pf::Observer obs(vanilla_pf::TIME_STEP, vanilla_pf::State{1.0});
  epf::ParticleFilter<vanilla_pf::State> pf(vanilla_pf::State{1.0}, vanilla_pf::PARTICLE_COUNT);
  pf.set_process_model<vanilla_pf::StateTransition>();
  pf.add_measurement_model<vanilla_pf::Observation>(obs);

  double mse = 0.0;
  std::vector<double> estimated_state(1, 1.0);
  for (std::size_t i = 1; i <= vanilla_pf::TIME_STEP; ++i, vanilla_pf::get_global_time().update_time_step()) {
    estimated_state.push_back(pf.sample().x);
  }

  auto const time_series_state     = obs.get_time_series();
  auto const generated_time_series = time_series_state | transform([](auto t_state) { return t_state.x; }) | to_vector;

  for_each(zip(estimated_state, generated_time_series), [&](auto t_zip) {
    auto const [estimated, current] = t_zip;
    mse += std::pow(current - estimated, 2);
    fmt::print("current value: {: f}, estimated value: {: f}, diff: {: f}\n", current, estimated, current - estimated);
  });

  mse /= (vanilla_pf::TIME_STEP);
  fmt::print("MSE PF: {: .7}\n", mse);

#if HAVE_MATPLOTCPP
  auto const x_axis = matplot::linspace(1, vanilla_pf::TIME_STEP);
  matplot::plot(x_axis, generated_time_series, "-", x_axis, estimated_state, "o");
  matplot::show();
#endif
}