#ifndef MEASUREMENT_HPP_
#define MEASUREMENT_HPP_

#include "epf/core/enum.hpp"
#include "epf/core/state.hpp"
#include "epf/util/exceptions.hpp"
#include <optional>
#include <type_traits>
#include <vector>

namespace epf {

/**
 *  @brief
 */
template <typename SensorData>
struct Measurement {
  virtual SensorData get_measurement() = 0;
  virtual bool data_ready()            = 0;

  Measurement()                              = default;
  Measurement(Measurement const&)            = default;
  Measurement& operator=(Measurement const&) = default;

  Measurement(Measurement&&) noexcept            = default;
  Measurement& operator=(Measurement&&) noexcept = default;

  virtual ~Measurement() = default;
};

template <typename State, typename Out = void>
struct MeasurementModel;

/**
 *  @brief  Abstract interface for measurement model, formalized as z_{k} = h_{k}(x_{k}, u_{k}, n_{k})
 *
 */
template <typename State>
struct MeasurementModel<State, void> {
  using StateVector = typename StateTraits<State>::ArithmeticType;

  /**
   *  @brief  This function assign weight to propagated state, according to latest measuremenet. Similar to
   *          ProcessModel, this function doesn't take care of measurement data, user should obtain it on their
   *          own.
   */
  virtual MeasurementResult update(std::vector<StateVector>& t_states, std::vector<double>& t_prev_weights) = 0;

  /**
   * @brief This function checks whether the data is ready for the measurement model. Particle filter may have faster
   *        sampling rate than some sensors, such as camera. Therefore, if the data is not ready yet, return false so
   *        that particle filter skip the sampling and resampling this time
   *
   * @return true if input is ready, false otherwise
   */
  [[nodiscard]] virtual bool data_ready() { return true; }

  /**
   *  @brief  This function generate random particle and assign weight to it according to latest measurement. This is
   *          used to add diversity in partcles so that particle filter can recover from incorrect state estimation. Not
   *          all algorithms require this to be implemented, as it is just one of the proposals to the problem mentioned
   *          above.
   */
  virtual std::pair<State, double> sample_state_from_latest_measurement() {
    throw NotImplementedError{"The algorithm require this function, but user didn't provide one"};
  }

  MeasurementModel()                                   = default;
  MeasurementModel(MeasurementModel const&)            = default;
  MeasurementModel& operator=(MeasurementModel const&) = default;

  MeasurementModel(MeasurementModel&&) noexcept            = default;
  MeasurementModel& operator=(MeasurementModel&&) noexcept = default;

  virtual ~MeasurementModel() = default;
};

template <typename State, typename Out>
struct MeasurementModel : MeasurementModel<State, void> {
  static_assert(satisfy_noise_traits<Out>::value);

  using StateVector       = typename StateTraits<State>::ArithmeticType;
  using OutputVector      = typename StateTraits<Out>::ArithmeticType;
  using OutputNoiseVector = typename NoiseTraits<Out>::ArithmeticType;
  using OutputNoiseCov    = typename NoiseTraits<Out>::NoiseCovType;

  void set_noise_covariance(OutputNoiseCov const& t_noise_cov) noexcept { this->noise_cov_ = t_noise_cov; }
  [[nodiscard]] OutputNoiseCov get_noise_covariance() const noexcept { return this->noise_cov_; }

  /**
   *
   */
  virtual std::vector<OutputVector> predict(std::vector<StateVector> const&, std::vector<OutputNoiseVector> const&) = 0;

  /**
   *
   */
  virtual Out get_latest_output() = 0;

  MeasurementModel()                                   = default;
  MeasurementModel(MeasurementModel const&)            = default;
  MeasurementModel& operator=(MeasurementModel const&) = default;

  MeasurementModel(MeasurementModel&&) noexcept            = default;
  MeasurementModel& operator=(MeasurementModel&&) noexcept = default;

  virtual ~MeasurementModel() = default;

 private:
  OutputNoiseCov noise_cov_{};
};

}  // namespace epf

#endif