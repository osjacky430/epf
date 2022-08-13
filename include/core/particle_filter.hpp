#ifndef PARTICLE_FILTER_HPP_
#define PARTICLE_FILTER_HPP_

#include "component/resampler/default.hpp"

#include "kdtree.hpp"
#include "measurement.hpp"
#include "process.hpp"

#include <cassert>
#include <memory>
#include <vector>

namespace epf {

template <typename T>
class ParticleAdapter {
 protected:
  virtual ~ParticleAdapter() = default;

  double weight_ = 0.0;
  T value_{};

 public:
  using value_type = T;

  ParticleAdapter()                           = default;
  ParticleAdapter(ParticleAdapter const&)     = default;
  ParticleAdapter(ParticleAdapter&&) noexcept = default;

  ParticleAdapter& operator=(ParticleAdapter const&) = default;
  ParticleAdapter& operator=(ParticleAdapter&&) noexcept = default;

  explicit ParticleAdapter(double const t_weight) : ParticleAdapter(T{}, t_weight) {}
  explicit ParticleAdapter(T&& t_v, double const t_weight = 0.0) : weight_{t_weight}, value_{std::forward<T>(t_v)} {
    assert(t_weight >= 0.0);
  }

  void set_weight(double const t_weight) noexcept {
    assert(t_weight >= 0.0);
    this->weight_ = t_weight;
  }

  void set_value(T&& t_value) noexcept { this->value_ = std::forward<T>(t_value); }

  [[nodiscard]] T weighted_value() const noexcept { return this->value_ * this->weight_; }

  [[nodiscard]] T value() const noexcept { return this->value_; }

  [[nodiscard]] double weight() const noexcept { return this->weight_; }
};

/**
 *
 */
template <typename ParticleType, typename ResampleStrategy = DefaultResample<ParticleType>>
class ParticleFilter : ResampleStrategy {
 protected:
  using ResampleStrategy::resample;

  auto get_all_measurement() const noexcept { return this->sensors_; }

  auto get_process_model() const noexcept { return this->motion_.get(); }

 public:
  static constexpr auto DEFAULT_MIN_PARTICLE_COUNTS = 100;
  static constexpr auto DEFAULT_MAX_PARTICLE_COUNTS = 2000;

  // template <typename Distribution>
  // static auto init_particles(Distribution t_distribution, std::size_t const t_size) noexcept {
  //   std::vector<ParticleType> ret_val(t_size);
  //   std::generate(ret_val.begin(), ret_val.end(), [&]() { return ParticleType::random_particle(t_distribution); });

  //   return ret_val;
  // }

  ParticleFilter() = default;

  ParticleFilter(ParticleFilter const&)     = default;
  ParticleFilter(ParticleFilter&&) noexcept = default;

  ParticleFilter& operator=(ParticleFilter const&) = default;
  ParticleFilter& operator=(ParticleFilter&&) noexcept = default;

  virtual ~ParticleFilter() = default;

  virtual ParticleType sample() noexcept = 0;

  template <typename Measurement, typename... InitArg>
  void add_measurement_model(InitArg&&... args) noexcept {
    this->sensors_.push_back(std::make_shared<Measurement>(std::forward<InitArg>(args)...));
  }

  template <typename Motion, typename... InitArg>
  void set_process_model(InitArg&&... args) noexcept {
    this->motion_ = std::make_unique<Motion>(std::forward<InitArg>(args)...);
  }

 private:
  std::unique_ptr<ProcessModel<ParticleType>> motion_{};
  std::vector<std::shared_ptr<MeasurementModel<ParticleType>>> sensors_{};  // TODO: change to unique_ptr
};

}  // namespace epf

#endif