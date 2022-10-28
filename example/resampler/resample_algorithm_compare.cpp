#include "epf/component/resampler/algorithm/multinomial.hpp"
#include "epf/component/resampler/algorithm/residual.hpp"
#include "epf/component/resampler/algorithm/stratified.hpp"
#include "epf/core/state.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <cstdlib>
#include <fmt/compile.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <random>
#include <range/v3/algorithm/generate.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>
#include <string_view>
#include <vector>

namespace {

inline constexpr auto PARTICLE_COUNT = 5;
inline constexpr auto SIM_TIME_STEP  = 100'000;

struct State {
  using ValueType = std::size_t;
  using Dimension = std::integral_constant<std::size_t, 1>;

  ValueType v_ = 0.0;
};

template <typename T>
struct Resampler final : public T {
  using T::resample_impl;
  [[nodiscard]] auto operator()(std::vector<typename T::StateVector> t_s, std::vector<double> t_w) {
    this->resample_impl(nullptr, t_s, t_w);
    return t_s;
  }

  void test_resampler(std::vector<typename T::StateVector> const& t_s, std::vector<double> const& t_w,
                      std::string_view const t_resampler_name) {
    using StorageType = Eigen::Matrix<double, Eigen::Dynamic, PARTICLE_COUNT>;
    StorageType count = StorageType::Zero(SIM_TIME_STEP, PARTICLE_COUNT);
    for (int idx : ranges::views::iota(0, SIM_TIME_STEP)) {
      auto const sv = this->operator()(t_s, t_w);
      for (auto const& val : sv) {
        ++count(idx, val[0]);
      }
    }

    auto const avg_occurence = count.colwise().mean();
    auto const std_dev       = (count.rowwise() - avg_occurence).array().pow(2).colwise().mean().array().sqrt();

    fmt::print(FMT_COMPILE("{} mean: {::.3f}\n"), t_resampler_name, avg_occurence);
    fmt::print(FMT_COMPILE("{} std : {::.3f}\n"), t_resampler_name, std_dev);
  }
};

}  // namespace

// todo: maybe we can generate unit test base on the result in the paper
int main(int /**/, char** /**/) {
  std::random_device rd{};
  std::mt19937 rng{rd()};
  std::uniform_real_distribution<> dist;

  auto const unormalized_weight = [&]() {
    std::vector<double> ret_val(::PARTICLE_COUNT);
    ranges::generate(ret_val, [&]() { return dist(rng); });
    return ret_val;
  }();

  auto const normalize = [weight_sum = ranges::accumulate(unormalized_weight, 0.0)](auto const t_v) {
    return t_v / weight_sum;
  };

  // todo: if normalized weight sum != 1, program will crash, need to find the reason
  auto const normalized_weight = unormalized_weight | ranges::views::transform(normalize) | ranges::to_vector;
  fmt::print(FMT_COMPILE("particle weights: {::.3f}\n"), normalized_weight);

  using StateVector = epf::StateTraits<::State>::ArithmeticType;
  auto const states = ranges::views::iota(0, ::PARTICLE_COUNT) |
                      ranges::views::transform([](auto t_v) { return StateVector{static_cast<double>(t_v)}; }) |
                      ranges::to_vector;

  ::Resampler<epf::Multinomial<::State>> m;
  ::Resampler<epf::Stratified<::State>> s;
  ::Resampler<epf::Residual<::State>> r;

  // This example is to reproduce the result in the paper: Particle Filters: A Hands-On Tutorial. Note that there is an
  // errata in the paper, which swaps the result of residual and systematic resampling. The output is commented here
  // for record:

  // Compare four resampling algorithms.
  // Input samples: [(0.366, 1), (0.354, 2), (0.119, 3), (0.058, 4), (0.103, 5)]
  // Testing MULTINOMIAL resampling
  // mean #occurrences: [1.83120169 1.76694233 0.59859401 0.28754712 0.51571484]
  // std  #occurrences: [1.07436274 1.06506912 0.72634451 0.51865512 0.67921346]
  // Testing RESIDUAL resampling
  // mean #occurrences: [1.83074169 1.7698323  0.59549405 0.29295707 0.51097489]
  // std  #occurrences: [0.77430313 0.7562715  0.69001343 0.51452177 0.65280762]
  // Testing STRATIFIED resampling
  // mean #occurrences: [1.82866171 1.76947231 0.59781402 0.2902771  0.51377486]
  // std  #occurrences: [0.3768043  0.61837095 0.63038944 0.45389019 0.49981022]
  // Testing SYSTEMATIC resampling
  // mean #occurrences: [1.8304417  1.77059229 0.59552404 0.2903671  0.51307487]
  // std  #occurrences: [0.3752443  0.42045191 0.49079034 0.45393176 0.49982902]

  // which is different from the paper:
  //
  // Resampling Alg.    Standard Deviations
  //   Multinomial    1.08  1.07  0.72  0.52  0.68
  //   Systematic     0.77  0.76  0.69  0.51  0.65
  //   Stratified     0.38  0.62  0.63  0.45  0.50
  //   Residual       0.38  0.42  0.49  0.46  0.50

  fmt::print("------------------------\n");
  m.test_resampler(states, normalized_weight, "multinomial");

  fmt::print("------------------------\n");
  s.test_resampler(states, normalized_weight, "stratified");

  fmt::print("------------------------\n");
  r.test_resampler(states, normalized_weight, "residual");

  return EXIT_SUCCESS;
}