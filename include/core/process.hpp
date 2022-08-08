#ifndef PROCESS_MODEL_HPP_
#define PROCESS_MODEL_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <random>

namespace epf {

enum class Prediction { Updated, NoUpdate };

/**
 *  @brief  Abstract interface for process model, a process model encodes prior knowledge on how state evolved over
 *          time. Formalized by mathematical model x_{k} = f_{k}(x_{k-1}, u_{k-1}, v_{k-1}), where x_{k} represents
 *          state at descrete time step k, u_{k-1}, v_{k-1} represents optional inputs, and i.i.d. noise, respectively
 *
 *  @tparam State state to predict
 */
template <typename State>
struct ProcessModel {
  virtual Prediction predict(std::vector<State>& /* t_prev_state */) = 0;

  ProcessModel()                    = default;
  ProcessModel(ProcessModel const&) = default;
  ProcessModel& operator=(ProcessModel const&) = default;

  ProcessModel(ProcessModel&&) noexcept = default;
  ProcessModel& operator=(ProcessModel&&) noexcept = default;

  virtual ~ProcessModel() = default;
};

template <typename InputType>
struct ControlInput {
  virtual InputType get_input() = 0;

  // TODO: learn clang-tidy warning, why do I need these, or can I ignore the warning
  ControlInput()                    = default;
  ControlInput(ControlInput const&) = default;
  ControlInput& operator=(ControlInput const&) = default;

  ControlInput(ControlInput&&) noexcept = default;
  ControlInput& operator=(ControlInput&&) noexcept = default;

  virtual ~ControlInput() = 0;
};

}  // namespace epf

#endif