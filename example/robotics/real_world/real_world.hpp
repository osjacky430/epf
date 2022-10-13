/**
 *  @file real_world.hpp
 */

#ifndef REAL_WORLD_HPP_
#define REAL_WORLD_HPP_

#include "epf/component/process_model/differential_drive.hpp"
#include <array>
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <random>
#include <range/v3/view/enumerate.hpp>
#include <vector>

namespace robotics {

struct Pose {
  double x_;
  double y_;
  double theta_;
};

struct Velocity {
  double linear_;
  double angular_;
};

struct Accel {
  double linear_;
  double angular_;
};

struct LandmarkData {
  std::array<double, 2> meas_;
  std::size_t landmark_idx_;

  LandmarkData(std::array<double, 2> const& t_meas, std::size_t t_idx) : meas_(t_meas), landmark_idx_(t_idx) {}
};

}  // namespace robotics

namespace real_world {

class WorldSim {
  using Path = std::vector<robotics::Pose>;

  double width_;
  double length_;
  std::size_t landmark_count_ = 0;

  double time_interval_;
  std::size_t sim_steps_;
  std::size_t current_step_ = 0;

  Path robot_path_{};
  std::vector<std::pair<double, double>> landmarks_ = {};

 public:
  WorldSim(double const t_width, double const t_length, std::size_t const t_landmark_count,
           std::size_t const t_sim_step, double const t_time_interval)
    : width_(t_width),
      length_(t_length),
      landmark_count_(t_landmark_count),
      landmarks_(landmark_count_),
      sim_steps_(t_sim_step),
      time_interval_(t_time_interval) {}

  [[nodiscard]] double get_width() const noexcept { return this->width_; }
  [[nodiscard]] double get_length() const noexcept { return this->length_; }

  void generate_landmarks(Path const& t_poses, double t_detection_range) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> theta_dist(0, boost::math::double_constants::two_pi);
    std::uniform_real_distribution<double> r_dist(0, t_detection_range);
    std::uniform_int_distribution<std::size_t> idx_dist(0, t_poses.size());

    for (auto& landmark : this->landmarks_) {
      auto const idx = idx_dist(rng);
      auto pose      = t_poses[idx];

      do {
        auto const r     = r_dist(rng);
        auto const theta = theta_dist(rng);
        pose.x_ += r * std::cos(theta);
        pose.y_ += r * std::sin(theta);
        landmark = {pose.x_, pose.y_};
      } while (!(0.0 <= pose.x_ and pose.x_ <= this->width_) or !(0.0 <= pose.y_ and pose.y_ <= this->length_));
    }
  }

  [[nodiscard]] auto get_landmarks() const noexcept { return this->landmarks_; }

  [[nodiscard]] auto get_robot_pose() const noexcept { return this->robot_path_.at(this->current_step_); }

  void set_robot_path(Path const& t_robot_path) noexcept { this->robot_path_ = t_robot_path; }

  void step() { ++this->current_step_; }

  [[nodiscard]] auto get_world_time() const noexcept {
    return static_cast<double>(this->current_step_) * this->time_interval_;
  }
};

/**
 * @brief DifferentialRobot class represent the real world motor control module, which takes velocity command, and
 *        update its poses in ODOM FRAME via encoder data
 *
 */
class DifferentialRobot {
  robotics::Pose current_pose_;
  double time_interval_;

  robotics::Velocity max_velocity_;
  robotics::Accel max_accel_;

  double linear_stddev_;
  double angular_stddev_;

  std::mt19937 rng{std::random_device()()};

 public:
  DifferentialRobot(double t_time_interval, robotics::Pose t_pose, robotics::Velocity t_max_vel,
                    robotics::Accel t_max_accel, double t_linear_stddev, double t_angular_stddev)
    : current_pose_(t_pose),
      time_interval_(t_time_interval),
      max_velocity_(t_max_vel),
      max_accel_(t_max_accel),
      linear_stddev_(t_linear_stddev),
      angular_stddev_(t_angular_stddev) {}

  void move(robotics::Velocity const& t_vel) noexcept {
    auto lin_vel_dist   = std::normal_distribution<>(0, this->linear_stddev_);
    auto ang_vel_dist   = std::normal_distribution<>(0, this->angular_stddev_);
    auto const lin_diff = (t_vel.linear_ + lin_vel_dist(this->rng)) * this->time_interval_;
    this->current_pose_.theta_ += (t_vel.angular_ + ang_vel_dist(this->rng)) * this->time_interval_;
    this->current_pose_.theta_ = epf::constraint_angle(this->current_pose_.theta_);
    this->current_pose_.x_ += lin_diff * std::cos(this->current_pose_.theta_);
    this->current_pose_.y_ += lin_diff * std::sin(this->current_pose_.theta_);
  }

  [[nodiscard]] robotics::Pose get_odom_data() const noexcept { return this->current_pose_; }
  [[nodiscard]] robotics::Velocity get_max_vel() const noexcept { return this->max_velocity_; }
  [[nodiscard]] robotics::Accel get_max_accel() const noexcept { return this->max_accel_; }
};

class LandmarkDetector {
  WorldSim* world_ = nullptr;

  double last_measure_timestamp_ = 0.0;
  double update_freq_            = 0.0;
  double detection_range_        = 0.0;
  double range_std_              = 0.0;
  double bearing_std_            = 0.0;

  std::mt19937 rng_{std::random_device{}()};
  std::normal_distribution<> range_dist_{0.0, range_std_};
  std::normal_distribution<> bearing_dist_{0.0, bearing_std_};

  using LandmarkCoor = std::pair<double, double>;

 public:
  explicit LandmarkDetector(WorldSim* const t_world, double const t_detection_range, double const t_range_std,
                            double const t_bearing_std, double const t_update_freq)
    : world_(t_world),
      detection_range_(t_detection_range),
      range_std_(t_range_std),
      bearing_std_(t_bearing_std),
      update_freq_(t_update_freq) {}

  [[nodiscard]] double get_detection_range() const noexcept { return this->detection_range_; }

  [[nodiscard]] auto detect() noexcept {
    this->last_measure_timestamp_ = this->world_->get_world_time();
    std::vector<robotics::LandmarkData> ret_val;
    auto const& landmark    = this->world_->get_landmarks();
    auto const current_pose = this->world_->get_robot_pose();
    for (auto [idx, position] : ranges::views::enumerate(landmark)) {
      auto const [x, y] = position;
      auto const x_diff = x - current_pose.x_;
      auto const y_diff = y - current_pose.y_;

      if (auto const distance = std::hypot(x_diff, y_diff); distance <= this->detection_range_) {
        auto const distance_noised = distance + this->range_dist_(this->rng_);
        auto const bearing_noised  = std::atan2(y_diff, x_diff) - current_pose.theta_ + this->bearing_dist_(this->rng_);
        ret_val.emplace_back(std::array{distance_noised, epf::constraint_angle(bearing_noised)}, idx);
      }
    }

    return ret_val;
  }

  [[nodiscard]] bool data_ready() const noexcept {
    return this->world_->get_world_time() - this->last_measure_timestamp_ >= this->update_freq_;
  }
};

struct Observer {
  using VelocityProfile = std::vector<robotics::Velocity>;
  using Path            = std::vector<robotics::Pose>;

  [[nodiscard]] static std::pair<VelocityProfile, Path> generate_path(double const t_time_interval,
                                                                      robotics::Pose const& t_initial_pose,
                                                                      WorldSim const& t_world,
                                                                      DifferentialRobot const& t_bot,
                                                                      std::size_t const t_sim_steps) {
    VelocityProfile vel_ret(t_sim_steps);
    Path path_ret(t_sim_steps);

    auto const max_vel   = t_bot.get_max_vel();
    auto const max_accel = t_bot.get_max_accel();

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> lin_accel_dist(0, max_accel.linear_);
    std::uniform_real_distribution<double> angular_accel_dist(0, max_accel.angular_);

    robotics::Velocity current_vel = {0.0, 0.0};
    robotics::Pose current_pose    = t_initial_pose;
    for (std::size_t i = 0; i < t_sim_steps; ++i) {
      // this may possibly stuck because we may be hitting the wall when we are at full speed, which leaves no space for
      // robot to deccelerate, i.e. no possible solution in this case, maybe we can make the map cyclic to solve this
      // problem (but then we need to modify differential drive model so that discontinuity makes sense, just like
      // rotation)
      do {
        auto const lin_accel       = lin_accel_dist(rng);
        auto const ang_accel       = angular_accel_dist(rng);
        auto const max_linear_inc  = lin_accel * t_time_interval;
        auto const max_angular_inc = ang_accel * t_time_interval;

        std::uniform_real_distribution<double> lin_vel_dist(-max_linear_inc, max_linear_inc);
        std::uniform_real_distribution<double> ang_vel_dist(-max_angular_inc, max_angular_inc);

        auto const lin_vel = std::clamp(current_vel.linear_ + lin_vel_dist(rng), -max_vel.linear_, max_vel.linear_);
        auto const ang_vel = std::clamp(current_vel.angular_ + ang_vel_dist(rng), -max_vel.angular_, max_vel.angular_);
        vel_ret[i]         = {lin_vel, ang_vel};

        // use x = delta_v * t for linear displacement instead of x = (v^2 - v_0^2) / (2 * a) since we are in descrete
        // world, and the latter equation indicates that within t_time_interval, the velocity increases linearly, which
        // makes calculating x, y coordinate more complicated since angular velocity also increases linearly. Using
        // former eq means that the robot can "jump" to desire velocity, less ideal, but makes coordinate calculation
        // way easier
        auto const line_diff = lin_vel * t_time_interval;
        auto const ang_diff  = ang_vel * t_time_interval;

        auto const theta = current_pose.theta_ + ang_diff;
        auto const x     = current_pose.x_ + line_diff * std::cos(theta);
        auto const y     = current_pose.y_ + line_diff * std::sin(theta);
        path_ret[i]      = {x, y, epf::constraint_angle(theta)};

        current_vel  = vel_ret[i];
        current_pose = path_ret[i];
      } while (!(0.0 <= current_pose.x_ and current_pose.x_ <= t_world.get_width()) or
               !(0.0 <= current_pose.y_ and current_pose.y_ <= t_world.get_length()));
    }

    return {vel_ret, path_ret};
  }
};

}  // namespace real_world

#endif