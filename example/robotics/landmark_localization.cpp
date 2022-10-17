#if HAVE_MATPLOTCPP
#include <matplot/matplot.h>
#endif

#include "epf/component/measurement_model/landmark.hpp"
#include "epf/component/misc/map.hpp"
#include "epf/component/process_model/differential_drive.hpp"
#include "epf/core/measurement.hpp"
#include "epf/core/particle_filter.hpp"
#include "real_world/real_world.hpp"
#include <boost/math/constants/constants.hpp>
#include <cstddef>
#include <fmt/format.h>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#include <utility>

namespace {

using namespace robotics;

inline constexpr double PI                      = boost::math::double_constants::pi;
inline constexpr double WORLD_WIDTH             = 100;
inline constexpr double WORLD_LENGTH            = 100;
inline constexpr std::size_t LANDMARK_COUNT     = 0;
inline constexpr double CAMERA_DETECTION_RANGE  = 10.0;
inline constexpr double TIME_STEP               = 0.5;
inline constexpr std::size_t SIM_STEPS          = 600;
inline constexpr double CAMERA_UPDATE_RATE      = 0.1;
inline constexpr double CAMERA_RANGE_VARIANCE   = 0.1;
inline constexpr double CAMERA_BEARING_VARIANCE = 0.01;

inline constexpr auto MOTOR_LINEAR_MOTION_STDDEV  = 0.1;
inline constexpr auto MOTOR_ANGULAR_MOTION_STDDEV = 0.1;

inline constexpr auto DIFFERENTIAL_DRIVE_ALPHA_ROT        = std::array{0.01, 0.01};
inline constexpr auto DIFFERENTIAL_DRIVE_ALPHA_TRANS      = std::array{0.01, 0.01};
inline constexpr auto DIFFERENTIAL_DRIVE_UPDATE_THRESHOLD = Pose{0.0, 0.0, 0.0};

inline constexpr Pose INITIAL_POSE      = {50, 50, 0};
inline constexpr Velocity MAX_VELOCITY  = {1.0, PI / 6.0};
inline constexpr Accel MAX_ACCELERATION = {0.25, PI / 36.0};

struct State {
  Pose p;

  struct Tag {};

  using TagType   = Tag;
  using ValueType = double;
  using Dimension = std::integral_constant<std::size_t, 3>;
};

/**
 *  @brief  This class adapts LandmarkData as SensorData for epf::LandmarkModel
 */
struct LandmarkMeasurement {
  using SignatureType = std::size_t;

  struct Data {
    LandmarkData data_;

    [[nodiscard]] double operator[](std::size_t const t_idx) const noexcept { return this->data_.meas_.at(t_idx); }
    [[nodiscard]] SignatureType signature() const noexcept { return this->data_.landmark_idx_; }
  };

  LandmarkMeasurement() = default;
  explicit LandmarkMeasurement(std::vector<LandmarkData> t_sensor_input)
    : data_(t_sensor_input | ranges::views::transform([](auto&& t_landmark) { return Data{t_landmark}; }) |
            ranges::to_vector) {}

  std::vector<Data> data_;

  [[nodiscard]] auto begin() const noexcept { return this->data_.begin(); }
  [[nodiscard]] auto end() const noexcept { return this->data_.end(); }
};

/**
 *  @brief  This class is the software interface of landmark sensor, call the API of the sensor provided by 3rd party
 *          library here
 */
struct CamMeasurement final : epf::Measurement<LandmarkMeasurement> {
  real_world::LandmarkDetector camera_;

  [[nodiscard]] LandmarkMeasurement get_measurement() override { return LandmarkMeasurement{camera_.detect()}; }

  [[nodiscard]] bool data_ready() override { return this->camera_.data_ready(); }

  explicit CamMeasurement(real_world::WorldSim* const t_world = nullptr)
    : camera_(t_world, CAMERA_DETECTION_RANGE, CAMERA_RANGE_VARIANCE, CAMERA_BEARING_VARIANCE, CAMERA_UPDATE_RATE) {}
};

struct OdomData {
  Pose p;

  [[nodiscard]] double operator[](std::size_t const t_idx) const { return std::array{p.x_, p.y_, p.theta_}.at(t_idx); }
};

struct OdomMeasurement final : epf::Measurement<OdomData> {
  real_world::DifferentialRobot* diff_drive_ = nullptr;

  [[nodiscard]] OdomData get_measurement() override { return OdomData{this->diff_drive_->get_odom_data()}; }

  [[nodiscard]] bool data_ready() override { return true; }

  explicit OdomMeasurement(real_world::DifferentialRobot* const t_diff_bot = nullptr) : diff_drive_(t_diff_bot) {}
};

struct LandmarkMapInterface final : epf::LandmarkMap2D<LandmarkMeasurement::SignatureType> {
  std::vector<std::pair<double, double>> landmarks_;

  PoseType get_landmark(LandmarkMeasurement::SignatureType const& t_idx) override {
    return PoseType{this->landmarks_[t_idx].first, this->landmarks_[t_idx].second, 0.0};
  }

  explicit LandmarkMapInterface(std::vector<std::pair<double, double>> t_landmarks)
    : landmarks_(std::move(t_landmarks)) {}
};

}  // namespace

namespace epf {

template <>
State from_state_vector<State>(Eigen::Vector3d const& t_v) {
  return State{Pose{t_v[0], t_v[1], t_v[2]}};
}

template <>
Eigen::Vector3d to_state_vector(const State& t_state) {
  return Eigen::Vector3d{t_state.p.x_, t_state.p.y_, t_state.p.theta_};
}

template <>
double& x_coor<State>(Eigen::Vector3d& t_v) {
  return t_v[0];
}

template <>
double& y_coor<State>(Eigen::Vector3d& t_v) {
  return t_v[1];
}

template <>
double& w_coor<State>(Eigen::Vector3d& t_v) {
  return t_v[2];
}

}  // namespace epf

int main(int /**/, char** /**/) {
  real_world::WorldSim world{WORLD_WIDTH, WORLD_LENGTH, LANDMARK_COUNT, SIM_STEPS, TIME_STEP};

  real_world::DifferentialRobot bot{
    TIME_STEP, INITIAL_POSE, MAX_VELOCITY, MAX_ACCELERATION, MOTOR_LINEAR_MOTION_STDDEV, MOTOR_ANGULAR_MOTION_STDDEV,
  };

  auto const profile = real_world::Observer::generate_path(TIME_STEP, INITIAL_POSE, world, bot, SIM_STEPS);
  world.generate_landmarks(profile.second, CAMERA_DETECTION_RANGE);

  auto const& landmarks = world.get_landmarks();
  LandmarkMapInterface map{landmarks};
  CamMeasurement sensor{&world};
  OdomMeasurement odom{&bot};
  epf::ParticleFilter<State> pf(State{INITIAL_POSE});

  auto* const landmark_model = pf.add_measurement_model<epf::LandmarkModel<LandmarkMeasurement, State>>();
  landmark_model->attach_sensor(&sensor);
  landmark_model->attach_map(&map);
  landmark_model->set_bearing_variance(CAMERA_BEARING_VARIANCE);
  landmark_model->set_range_variance(CAMERA_RANGE_VARIANCE);

  auto* const diff_drive_model = pf.set_process_model<epf::DifferentialDrive<OdomData, State>>();
  diff_drive_model->set_pose(OdomData{INITIAL_POSE});
  diff_drive_model->set_alpha_rot(DIFFERENTIAL_DRIVE_ALPHA_ROT);
  diff_drive_model->set_alpha_trans(DIFFERENTIAL_DRIVE_ALPHA_TRANS);
  diff_drive_model->attach_measurement(&odom);
  diff_drive_model->set_update_threshold(OdomData{DIFFERENTIAL_DRIVE_UPDATE_THRESHOLD});

  world.set_robot_path(profile.second);
  if constexpr (LANDMARK_COUNT > 0) {
    fmt::print("Pose generation done, generate landmark according to generated pose\n");
    for (auto const landmark : world.get_landmarks()) {
      fmt::print("\tlandmark: ({}, {})\n", landmark.first, landmark.second);
    }
  }

#if HAVE_MATPLOTCPP
  namespace rv = ranges::views;

  auto const landmarks_x = landmarks | rv::transform([](auto const& t_v) { return t_v.first; }) | ranges::to_vector;
  auto const landmarks_y = landmarks | rv::transform([](auto const& t_v) { return t_v.second; }) | ranges::to_vector;

  auto const robot_pose_x = profile.second | rv::transform([](auto const& t_v) { return t_v.x_; }) | ranges::to_vector;
  auto const robot_pose_y = profile.second | rv::transform([](auto const& t_v) { return t_v.y_; }) | ranges::to_vector;
  auto map_plt            = matplot::subplot(3, 2, 0);
  map_plt->hold(true);
  if constexpr (LANDMARK_COUNT > 0) {
    map_plt->plot(landmarks_x, landmarks_y, "o");
  }
  map_plt->plot(robot_pose_x, robot_pose_y);
  map_plt->xlim({0.0, world.get_width()});
  map_plt->ylim({0.0, world.get_length()});

  std::vector<State> estimated_state;
#endif

  for (std::size_t i = 0; i < SIM_STEPS; ++i, world.step()) {
    auto const& cmd_vel           = profile.first[i];
    auto const& pose_ground_truth = profile.second[i];

    bot.move(cmd_vel);

    State p = pf.sample();
#if HAVE_MATPLOTCPP
    estimated_state.push_back(p);
#endif
  }

#if HAVE_MATPLOTCPP
  auto x_estimation_plt  = matplot::subplot(3, 2, 1);
  auto const x_truth     = profile.second | rv::transform([](auto const& t_v) { return t_v.x_; }) | ranges::to_vector;
  auto const x_estimated = estimated_state | rv::transform([](auto& t_v) { return t_v.p.x_; }) | ranges::to_vector;
  x_estimation_plt->hold(true);
  x_estimation_plt->plot(x_truth);
  x_estimation_plt->plot(x_estimated);

  auto y_estimation_plt  = matplot::subplot(3, 2, 2);
  auto const y_truth     = profile.second | rv::transform([](auto const& t_v) { return t_v.y_; }) | ranges::to_vector;
  auto const y_estimated = estimated_state | rv::transform([](auto& t_v) { return t_v.p.y_; }) | ranges::to_vector;
  y_estimation_plt->hold(true);
  y_estimation_plt->plot(y_truth);
  y_estimation_plt->plot(y_estimated);

  auto theta_estimation_plt = matplot::subplot(3, 2, 3);
  auto const theta     = profile.second | rv::transform([](auto const& t_v) { return t_v.theta_; }) | ranges::to_vector;
  auto const theta_est = estimated_state | rv::transform([](auto& t_v) { return t_v.p.theta_; }) | ranges::to_vector;
  theta_estimation_plt->hold(true);
  theta_estimation_plt->plot(theta);
  theta_estimation_plt->plot(theta_est);

  auto linear_v_estimation_plt = matplot::subplot(3, 2, 4);
  auto const v_truth = profile.first | rv::transform([](auto const& t_v) { return t_v.linear_; }) | ranges::to_vector;
  linear_v_estimation_plt->hold(true);
  linear_v_estimation_plt->plot(v_truth);

  auto angular_v_estimation_plt = matplot::subplot(3, 2, 5);
  auto const w_truth = profile.first | rv::transform([](auto const& t_v) { return t_v.angular_; }) | ranges::to_vector;
  angular_v_estimation_plt->hold(true);
  angular_v_estimation_plt->plot(w_truth);
  matplot::show();
#endif
}