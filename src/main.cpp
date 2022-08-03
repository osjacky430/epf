#include "kdtree.hpp"
#include "particle_filter.hpp"
#include "robotics/amcl.hpp"
#include "robotics/models/differential_drive.hpp"
#include "robotics/models/laser_beam.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <utility>

// TODO: separate weight and pose, (and separate pose to translation and rotation, if possible)
class Pose {
 public:
  static constexpr auto Dimension = 3;

  using PoseType       = Eigen::Array3d;
  using CoordinateType = Eigen::Array2d;
  using AngleType      = double;

  Pose() = default;
  Pose(double const t_x, double const t_y, double const t_w) : pose_{t_x, t_y, t_w} {}
  explicit Pose(PoseType t_pose) : pose_{std::move(t_pose)} {}

  inline double get(std::size_t const t_idx) const noexcept { return this->pose_[static_cast<long>(t_idx)]; }

  inline void set(std::size_t const t_idx, double const t_value) noexcept {
    this->pose_[static_cast<long>(t_idx)] = t_value;
  }

  [[nodiscard]] Pose transform_coordinate(Pose const& t_relative) const noexcept {
    auto const& relative_pose = t_relative.pose_;
    auto const translation    = Eigen::Vector2d(relative_pose[0], relative_pose[1]);
    auto const rotation       = Eigen::Rotation2Dd(relative_pose[2]);
    auto const origin         = Eigen::Array2d{this->pose_[0], this->pose_[1]};
    auto transformed          = origin + (rotation * translation).array();
    auto const angle          = this->pose_[2] + relative_pose[2];
    return Pose{Eigen::Array3d{transformed[0], transformed[1], std::atan2(std::sin(angle), std::cos(angle))}};
  }

 private:
  PoseType pose_{0, 0, 0};
};

struct LaserBeamMeas {
  std::vector<std::pair<double, double>> raw_measurements_;
};

struct Cell {
  bool occupied;
};

template <typename CellType, typename ParticleType>
class Map : public particle_filter::MapBase<ParticleType> {
 public:
  using MapData = std::vector<CellType>;

  [[nodiscard]] std::vector<std::array<double, 3>> get_free_cells() const noexcept override { return {}; }

  Map() = default;
  explicit Map(MapData const& t_map_content) : size_(t_map_content.size()), content_(t_map_content) {}

 private:
  std::size_t size_ = 0;
  MapData content_;
};

using PoseParticle = particle_filter::ParticleAdapter<Pose>;

int main(int /*argc*/, char** /*argv*/) {
  amcl::AMCL2D<PoseParticle> p;

  Map<Cell, PoseParticle> m;
  p.add_sensor<particle_filter::LaserBeamModel<LaserBeamMeas, PoseParticle>>();
  p.set_motion_model<particle_filter::Differential<PoseParticle>>();

  // while (true) {
  p.sample(m);
  // }

  std::cout << "Hello world\n";
  return EXIT_SUCCESS;
}