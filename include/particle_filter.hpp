#ifndef PARTICLE_FILTER_HPP_
#define PARTICLE_FILTER_HPP_

#include "kdtree.hpp"

#include "motion_model.hpp"
#include "sensor_model.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace particle_filter {

struct PivotComp;

template <typename T, typename PC = PivotComp>
struct ParticleAdapter {
  using ValueType = T;
  using PivotComp = PC;

  friend PivotComp;

  static constexpr auto Dimension = T::Dimension;

  ParticleAdapter() = default;
  explicit ParticleAdapter(T const& t_value, double const t_weight = 0.0) : value_{t_value}, weight_{t_weight} {}

  [[nodiscard]] T value() const noexcept { return this->value_; }
  [[nodiscard]] double weight() const noexcept { return this->weight_; }

  [[nodiscard]] T& value() noexcept { return this->value_; }
  [[nodiscard]] double& weight() noexcept { return this->weight_; }

  auto operator*(double const t_scale) const noexcept {
    T tmp{};
    for (std::size_t idx = 0; idx < Dimension; ++idx) {
      tmp.set(idx, this->value_.get(idx) * t_scale);
    }

    return ParticleAdapter<T>{this->value_, this->weight_};
  }

  auto operator+(ParticleAdapter<T> const& t_add) const noexcept {
    T tmp{};
    for (std::size_t idx = 0; idx < Dimension; ++idx) {
      tmp.set(idx, this->value_.get(idx) + t_add.value_.get(idx));
    }

    return ParticleAdapter<T>{tmp, this->weight_};
  }

  ParticleAdapter<T>& operator+=(ParticleAdapter<T> const& t_add) noexcept {
    for (std::size_t idx = 0; idx < Dimension; ++idx) {
      this->value_.set(idx, this->value_.get(idx) + t_add.value_.get(idx));
    }

    return *this;
  }

  ParticleAdapter<T>& operator/=(double const t_scale) noexcept {
    for (std::size_t idx = 0; idx < Dimension; ++idx) {
      this->value_.set(idx, this->value_.get(idx) / t_scale);
    }

    return *this;
  }

  auto operator/(ParticleAdapter<T> const& t_divisor) const noexcept {
    T tmp{};
    for (std::size_t idx = 0; idx < Dimension; ++idx) {
      tmp.set(idx, this->value_.get(idx) / t_divisor.value_.get(idx));
    }

    return ParticleAdapter<T>{tmp, this->weight_};
  }

  auto operator-(ParticleAdapter<T> const& t_lhs) const noexcept {
    T tmp{};
    for (std::size_t idx = 0; idx < Dimension; ++idx) {
      tmp.set(idx, this->value_.get(idx) - t_lhs.value_.get(idx));
    }

    return ParticleAdapter<T>{tmp, this->weight_};
  }

  bool operator==(ParticleAdapter<T> const& t_rhs) const noexcept {
    for (std::size_t idx = 0; idx < Dimension; ++idx) {
      if (t_rhs.value_.get(idx) != this->value_.get(idx)) {
        return false;
      }
    }

    return true;
  }

  double operator[](std::size_t const t_idx) const noexcept { return this->value_.get(t_idx); }

  void update(ParticleAdapter<T> const& t_pt) noexcept { this->weight_ += t_pt.weight(); }

 private:
  T value_;
  double weight_ = 0.0;

  std::size_t pivot_idx_ = T::Dimension;
  double pivot_value_    = 0;
};

struct PivotComp {
  template <typename T>
  bool operator()(ParticleAdapter<T>& t_lhs, ParticleAdapter<T> const& t_rhs) const noexcept {
    if (t_lhs.pivot_idx_ < T::Dimension) {
      return t_lhs.pivot_value_ < t_rhs.value_.get(t_lhs.pivot_idx_);
    }

    double max_diff          = 0.0;
    std::size_t max_diff_idx = 0;
    for (std::size_t idx = 0; idx < ParticleAdapter<T>::Dimension; ++idx) {
      auto const diff = t_lhs.value_.get(idx) - t_rhs.value_.get(idx);
      if (std::abs(diff) > max_diff) {
        max_diff     = std::abs(diff);
        max_diff_idx = idx;
      }
    }

    t_lhs.pivot_idx_   = max_diff_idx;
    t_lhs.pivot_value_ = (t_lhs.value_.get(max_diff_idx) + t_rhs.value_.get(max_diff_idx)) / 2.0;  // mean split

    return t_lhs.pivot_value_ < t_rhs.value_.get(max_diff_idx);
  }
};

// only importance resampling
template <typename ParticleType>
struct DefaultResample {
  std::mt19937 rng_ = std::mt19937(std::random_device{}());

  std::vector<ParticleType> resample(std::vector<ParticleType> const& t_previous_particles) const noexcept {
    auto const cumulative_weight = [&]() {
      std::vector<double> ret_val(t_previous_particles.size() + 1, 0);
      for (std::size_t i = 0; i < t_previous_particles.size(); ++i) {
        ret_val[i + 1] = t_previous_particles[i].weight() + ret_val[i];
      }

      return ret_val;
    }();

    std::vector<ParticleType> ret_val(t_previous_particles.size());
    if (cumulative_weight.back() == 0.0) {  // cumulative_weight.back() == weight sum
      return ret_val;
    }

    double total_weight = 0.0;
    std::uniform_real_distribution<> uniform_dist(0.0, 1.0);

    std::generate(ret_val.begin(), ret_val.end(), [&]() {
      auto const iter = std::lower_bound(cumulative_weight.begin(), cumulative_weight.end(), uniform_dist(this->rng_));
      auto chosen_particle = t_previous_particles[iter - cumulative_weight.begin()];
      total_weight += chosen_particle.weight();
      return chosen_particle;
    });

    std::for_each(ret_val.begin(), ret_val.end(), [=](auto& t_particle) { t_particle.weight() /= total_weight; });
    return ret_val;
  }
};

/**
 *
 */
template <typename ParticleType, typename ResampleStrategy = DefaultResample<ParticleType>>
class ParticleFilter : ResampleStrategy {
 protected:
  using ResampleStrategy::resample;

  auto get_all_sensors() const noexcept { return this->sensors_; }

  auto get_motion_model() const noexcept { return this->motion_.get(); }

 public:
  static constexpr auto DEFAULT_MIN_PARTICLE_COUNTS = 100;
  static constexpr auto DEFAULT_MAX_PARTICLE_COUNTS = 2000;

  template <typename Distribution>
  static auto init_particles(Distribution t_distribution, std::size_t const t_size) noexcept {
    std::vector<ParticleType> ret_val(t_size);
    std::generate(ret_val.begin(), ret_val.end(), [&]() { return ParticleType::random_particle(t_distribution); });

    return ret_val;
  }

  ParticleFilter() = default;

  ParticleFilter(ParticleFilter const&)     = default;
  ParticleFilter(ParticleFilter&&) noexcept = default;

  ParticleFilter& operator=(ParticleFilter const&) = default;
  ParticleFilter& operator=(ParticleFilter&&) noexcept = default;

  virtual ~ParticleFilter() = default;

  virtual ParticleType sample(particle_filter::MapBase<ParticleType>& t_map) noexcept = 0;

  template <typename Sensor, typename... InitArg>
  void add_sensor(InitArg&&... args) noexcept {
    this->sensors_.push_back(std::make_shared<Sensor>(std::forward<InitArg>(args)...));
  }

  template <typename Motion, typename... InitArg>
  void set_motion_model(InitArg&&... args) noexcept {
    this->motion_ = std::make_unique<Motion>(std::forward<InitArg>(args)...);
  }

 private:
  std::unique_ptr<MotionModel<ParticleType>> motion_{};
  std::vector<std::shared_ptr<SensorModel<ParticleType>>> sensors_{};  // TODO: change to unique_ptr
};

}  // namespace particle_filter

#endif