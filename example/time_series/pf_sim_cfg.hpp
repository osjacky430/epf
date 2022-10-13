#ifndef TIME_SERIES_SIM_CFG_
#define TIME_SERIES_SIM_CFG_

#include "epf/core/enum.hpp"
#include "epf/core/measurement.hpp"
#include "epf/core/process.hpp"
#include "epf/core/state.hpp"
#include "time_series.hpp"
#include <boost/math/constants/constants.hpp>
#include <cstddef>
#include <random>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/zip.hpp>
#include <tuple>
#include <type_traits>

namespace time_series {

inline constexpr auto PARTICLE_COUNT = 200;
inline constexpr auto TIME_STEP      = 60;
inline constexpr auto PI             = boost::math::double_constants::pi;

struct Timer {
  double time_step_;
};

template <typename TimeSeries>
struct Observer {
  typename TimeSeries::Output current_meas_{};
};

template <typename TimeSeries>
struct StateTransition : public epf::ProcessModel<typename TimeSeries::State> {
  Timer* timer_ = nullptr;

  std::mt19937 rng_{std::random_device{}()};
  std::normal_distribution<> norm_dist_{TimeSeries::STATE_NOISE_MEAN, std::sqrt(TimeSeries::STATE_NOISE_COV)};

  using State            = typename TimeSeries::State;
  using Base             = epf::ProcessModel<State>;
  using StateVector      = typename Base::StateVector;
  using StateNoiseVector = typename Base::StateNoiseVector;

 public:
  explicit StateTransition(Timer* const t_timer) : timer_(t_timer) {}

  [[nodiscard]] epf::Prediction predict(std::vector<StateVector>& t_state) override {
    std::for_each(t_state.begin(), t_state.end(), [&](auto& t_val) mutable {
      auto const time      = static_cast<std::size_t>(this->timer_->time_step_);
      auto const state_vec = epf::from_state_vector<State>(t_val);
      auto const predicted = TimeSeries::state_transition(state_vec, time, this->norm_dist_(this->rng_));
      t_val                = epf::to_state_vector<State>(predicted);
    });

    return epf::Prediction::Updated;
  }

  [[nodiscard]] epf::Prediction predict(std::vector<StateVector>& t_state,
                                        std::vector<StateNoiseVector> const& t_noise) override {
    ranges::for_each(ranges::views::zip(t_state, t_noise), [this](auto t_val) {
      auto [state, noise] = t_val;

      auto const time      = static_cast<std::size_t>(this->timer_->time_step_);
      auto const predicted = TimeSeries::state_transition(epf::from_state_vector<State>(state), time, noise[0]);
      state                = epf::to_state_vector(predicted);
    });

    return epf::Prediction::Updated;
  }

  [[nodiscard]] double calculate_probability(StateVector const& t_cur, StateVector const& t_prev) override {
    auto const time      = static_cast<std::size_t>(this->timer_->time_step_);
    auto const predicted = TimeSeries::state_transition(epf::from_state_vector<State>(t_prev), time);
    auto const noise_var = t_cur - epf::to_state_vector<State>(predicted);

    return std::exp(-std::pow(noise_var[0] - TimeSeries::STATE_NOISE_MEAN, 2) / (2 * TimeSeries::STATE_NOISE_COV));
  }
};

template <typename TimeSeries>
class ObserveOutput final : public epf::MeasurementModel<typename TimeSeries::State, typename TimeSeries::Output> {
  Timer* timer_              = nullptr;
  Observer<TimeSeries>* obs_ = nullptr;

  std::mt19937 rng_{std::random_device{}()};
  std::normal_distribution<> dist_{0, std::sqrt(TimeSeries::OBSERVATION_NOISE)};

  using State             = typename TimeSeries::State;
  using Output            = typename TimeSeries::Output;
  using Base              = epf::MeasurementModel<State, Output>;
  using StateVector       = typename Base::StateVector;
  using OutputVector      = typename Base::OutputVector;
  using OutputNoiseVector = typename Base::OutputNoiseVector;

 public:
  ObserveOutput(Timer* const t_timer, Observer<TimeSeries>* const t_obs) : timer_(t_timer), obs_(t_obs) {}

  [[nodiscard]] epf::MeasurementResult update(std::vector<StateVector>& t_states,
                                              std::vector<double>& t_weight) override {
    auto const observation = this->obs_->current_meas_;

    for (auto [state, weight] : ranges::views::zip(t_states, t_weight)) {
      auto const predict_obs =
        TimeSeries::observation(epf::from_state_vector<State>(state),
                                static_cast<std::size_t>(this->timer_->time_step_), this->dist_(this->rng_));

      weight *= std::exp(-std::pow(observation.x_ - predict_obs.x_, 2) / (2 * TimeSeries::OBSERVATION_NOISE));
    }

    auto const weight_sum = ranges::accumulate(t_weight, 0.0);
    auto const normalize  = [weight_sum, particle_size = t_weight.size()](auto& t_val) {
      if (weight_sum == 0) {
        t_val = 1.0 / static_cast<double>(particle_size);
        return;
      }

      t_val /= weight_sum;
    };
    ranges::for_each(t_weight, normalize);

    return epf::MeasurementResult::Updated;
  }

  [[nodiscard]] std::vector<OutputVector> predict(std::vector<StateVector> const& t_pt,
                                                  std::vector<OutputNoiseVector> const& t_noise) override {
    std::vector<OutputVector> ret_val(t_pt.size());

    for (auto [state, noise, obs] : ranges::views::zip(t_pt, t_noise, ret_val)) {
      obs = epf::to_state_vector<Output>(TimeSeries::observation(
        epf::from_state_vector<State>(state), static_cast<std::size_t>(this->timer_->time_step_), noise[0]));
    }

    return ret_val;
  }

  Output get_latest_output() override { return Output{this->obs_->current_meas_}; }
};

template <typename TimeSeries>
struct Simulator final : public TimeSeries {
  using TimeSeries::observation;
  using TimeSeries::state_transition;

  using State = typename TimeSeries::State;

  State initial_cond_;
  std::size_t total_step_;
  Timer timer_;
  Observer<TimeSeries> obs_;

  explicit Simulator(State const& t_initial_cond, std::size_t const t_total_sim_time, std::size_t t_start = 0)
    : TimeSeries(t_initial_cond, t_total_sim_time, t_start),
      initial_cond_(t_initial_cond),
      total_step_(t_total_sim_time),
      timer_{static_cast<double>(t_start)} {}

  template <typename TimeStepAction, typename Cb>
  void simulate(TimeStepAction&& t_time_step_action, Cb&& t_time_step_callback) {
    for (std::size_t i = 1; i < this->total_step_; ++i) {
      ++this->timer_.time_step_;
      this->obs_.current_meas_ = this->get_measurement(this->timer_.time_step_);
      t_time_step_callback(t_time_step_action(i));
    }
  }

  using ProcessModel = StateTransition<TimeSeries>;
  using OutputModel  = ObserveOutput<TimeSeries>;

  template <typename PF>
  auto create_particle_filter(std::size_t const t_max_particle) noexcept {
    PF f{this->initial_cond_, t_max_particle};
    auto* pm = f.template set_process_model<ProcessModel>(&this->timer_);
    auto* mm = f.template add_measurement_model<OutputModel>(&this->timer_, &this->obs_);
    return std::tuple{std::move(f), pm, mm};
  }
};

}  // namespace time_series

#endif