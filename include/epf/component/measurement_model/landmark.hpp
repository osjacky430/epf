#ifndef LANDMARK_HPP_
#define LANDMARK_HPP_

#include "epf/component/misc/map.hpp"
#include "epf/core/enum.hpp"
#include "epf/core/measurement.hpp"
#include "epf/core/state.hpp"
#include "epf/util/math.hpp"
#include <algorithm>
#include <boost/concept/usage.hpp>
#include <cmath>
#include <optional>
#include <range/v3/view/zip.hpp>

namespace epf {

template <typename SensorData>
struct LandmarkSensorDataConcept {  // NOLINT(*-special-member-functions)
  BOOST_CONCEPT_USAGE(LandmarkSensorDataConcept) {
    SensorData s{};

    for (auto const detected_landmark : s) {  // SensorData should have begin() and end()
      [[maybe_unused]] typename SensorData::SignatureType signature = detected_landmark.signature();
      [[maybe_unused]] auto const range   = detected_landmark[0];  // value_type should have [] operator
      [[maybe_unused]] auto const bearing = detected_landmark[1];
    }
  }
};

struct LandmarkModelParam {
  double range_var_;
  double bearing_var_;
};

template <typename SensorData, typename State>
class LandmarkModel final : public MeasurementModel<State> {
  using StateVector = typename StateTraits<State>::ArithmeticType;

  BOOST_CONCEPT_ASSERT((LandmarkSensorDataConcept<SensorData>));

  Measurement<SensorData>* sensor_ = nullptr;
  SensorData current_measurement_{};

  LandmarkMap2D<typename SensorData::SignatureType>* map_ = nullptr;

  double range_var_   = 0.0;
  double bearing_var_ = 0.0;

 public:
  LandmarkModel() = default;
  explicit LandmarkModel(LandmarkModelParam const& t_param, Measurement<SensorData>* const t_sensor)
    : range_var_(t_param.range_var_), bearing_var_(t_param.bearing_var_), sensor_(t_sensor) {}

  void attach_map(LandmarkMap2D<typename SensorData::SignatureType>* const t_map) noexcept { this->map_ = t_map; }
  void attach_sensor(Measurement<SensorData>* const t_sensor) noexcept { this->sensor_ = t_sensor; }

  void set_range_variance(double const t_var) noexcept { this->range_var_ = t_var; }
  void set_bearing_variance(double const t_var) noexcept { this->bearing_var_ = t_var; }

  [[nodiscard]] bool data_ready() override { return this->sensor_->data_ready(); }

  MeasurementResult update(std::vector<StateVector>& t_states, std::vector<double>& t_weight) noexcept override {
    double weight_sum      = 0.0;
    SensorData income_meas = this->sensor_->get_measurement();

    // nothing detected is equivalent to no measurement, we are not recalclating the weight, therefore the resampling
    // step is meaningless
    if (std::distance(income_meas.begin(), income_meas.end()) == 0) {
      return MeasurementResult::NoMeasurement;
    }

    for (auto [state, weight] : ranges::views::zip(t_states, t_weight)) {
      for (auto const& detected_landmark : income_meas) {
        auto const landmark_coor = this->map_->get_landmark(detected_landmark.signature());

        auto const x_diff              = landmark_coor[0] - state[0];
        auto const y_diff              = landmark_coor[1] - state[1];
        auto const theoretical_range   = std::hypot(x_diff, y_diff);
        auto const theoretical_bearing = epf::constraint_angle(std::atan2(y_diff, x_diff) - state[2]);

        auto const range_diff   = theoretical_range - detected_landmark[0];
        auto const bearing_diff = epf::constraint_angle(theoretical_bearing - detected_landmark[1]);

        auto const range_prob   = (std::pow(range_diff, 2) / (-2 * this->range_var_));
        auto const bearing_prob = (std::pow(bearing_diff, 2) / (-2 * this->bearing_var_));

        weight *= std::exp(range_prob + bearing_prob);
      }

      weight_sum += weight;
    }

    std::for_each(t_weight.begin(), t_weight.end(), [weight_sum](auto& t_val) { t_val /= weight_sum; });

    return MeasurementResult::Updated;
  }
};

}  // namespace epf

#endif