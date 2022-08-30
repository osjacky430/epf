#ifndef PROCESS_MODEL_HPP_
#define PROCESS_MODEL_HPP_

#include "enum.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <random>

namespace epf {

/**
 *  @brief  Abstract interface for process model, a process model encodes prior knowledge on how state evolved over
 *          time. Formalized by mathematical model x_{k} = f_{k}(x_{k-1}, u_{k-1}, v_{k-1}), where x_{k} represents
 *          state at descrete time step k, u_{k-1}, v_{k-1} represents optional inputs, and i.i.d. noise, respectively
 *
 *  @tparam State state to predict
 */
template <typename State>
struct ProcessModel {
  /**
   *  @brief  This function propagates state according to state transition model, notice that this function only takes
   *          x_{k-1}, without input, u_{k-1}. Due to the fact that particle filter has no idea how to obtain
   *          the input, user should do it on their own in this function. Also, noise, v_{k-1}, is not passed as well
   *          since particle filter has no idea how a process model its noise, could be normal gausian rv, or gamma rv,
   *          etc.
   *
   *  @param  t_prev_state  Previous state
   */
  virtual Prediction predict(std::vector<State>& /* t_prev_state */) = 0;

  /**
   *  @brief  This function propagates state according to state transition model, the only difference between this one
   *          and the previous one is that the noise parameter. This can be useful for certain algorithm that takes
   *          noise into consideration, such as unscented particle filter, see [1] for more detail.
   *
   *  [1] Rudolph van der Merwe, Arnaud Doucet, Nando de Freitas, Eric Wan. August 16, 2000, THE UNSCENTED PARTICLE
   *      FILTER
   *
   *  @param  t_prev_state  Previous state
   *  @param  t_artificial_noise  Noise generate by the algorithm
   *
   *  @todo   this should be pure virtual function (?)
   */
  virtual Prediction predict(std::vector<State>& t_prev_state, std::vector<State> const& /* t_artificial_noise */) {
    return this->predict(t_prev_state);
  }

  /**
   *  @brief  This function calculate the probability to reach to current state given previous state, and input,
   *          i.e. p(x_{t}|x_{t-1},u_{t}). As mentioned above, user should obtain the input on their own. This might
   *          be needed if one chooses different propotional distribution as default one, i.e. p(x_{t}|x_{t-1},u_{t})
   *
   *  @todo   Mark it as pure virtual function in the future
   */
  [[nodiscard]] virtual double calculate_probability(State const& /* t_curr */, State const& /* t_prev */) noexcept {
    return 1.0;
  }

  ProcessModel()                    = default;
  ProcessModel(ProcessModel const&) = default;
  ProcessModel& operator=(ProcessModel const&) = default;

  ProcessModel(ProcessModel&&) noexcept = default;
  ProcessModel& operator=(ProcessModel&&) noexcept = default;

  virtual ~ProcessModel() = default;
};

}  // namespace epf

#endif