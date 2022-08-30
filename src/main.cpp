#include "app/amcl.hpp"
#include "component/measurement_model/laser_beam.hpp"
#include "component/process_model/differential_drive.hpp"
#include "map.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <utility>

struct LaserBeamMeas {
  std::vector<std::pair<double, double>> raw_measurements_;

  [[nodiscard]] bool operator!=(LaserBeamMeas const& t_rhs) const noexcept {
    return this->raw_measurements_ != t_rhs.raw_measurements_;
  }

  [[nodiscard]] bool operator==(LaserBeamMeas const& t_rhs) const noexcept {
    return this->raw_measurements_ == t_rhs.raw_measurements_;
  }
};

struct Cell {
  bool occupied;
};

template <typename CellType, typename ParticleType>
class Map : public epf::MapBase<ParticleType> {
 public:
  using MapData = std::vector<CellType>;

  [[nodiscard]] std::vector<std::array<double, 3>> get_free_cells() const noexcept override { return {}; }

  Map() = default;
  explicit Map(MapData const& t_map_content) : size_(t_map_content.size()), content_(t_map_content) {}

 private:
  std::size_t size_ = 0;
  MapData content_;
};

struct Odom {
  [[nodiscard]] double operator[](std::size_t const /**/) const noexcept { return 1.0; }
};

int main(int /*argc*/, char** /*argv*/) {
  // epf::amcl::AMCL2D<> p;

  // Map<Cell, PoseParticle> m;

  // p.add_measurement_model<epf::LaserBeamModel<LaserBeamMeas, epf::amcl::Particle>>();
  // p.set_process_model<epf::Differential<Odom, epf::amcl::Particle>>();

  // while (true) {
  // p.sample(m);
  // }

  std::cout << "Hello world\n";
  return EXIT_SUCCESS;
}