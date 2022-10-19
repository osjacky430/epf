#ifndef COMMON_PF_HPP_
#define COMMON_PF_HPP_

#include "epf/component/importance_sampler/default.hpp"
#include "epf/component/importance_sampler/ekf.hpp"
#include "epf/component/importance_sampler/ukf.hpp"
#include "epf/component/resampler/algorithm/multinomial.hpp"
#include "epf/component/resampler/scheme/always.hpp"
#include "epf/core/particle_filter.hpp"

namespace epf {

template <typename State>
using VanillaPF = ParticleFilter<State, DefaultSampler<State>, MultinomialResampler<State, AlwaysScheme>>;

template <typename State, typename Output, typename Resampler = MultinomialResampler<State, AlwaysScheme>>
using UnscentedPF = ParticleFilter<State, UKFSampler<State, Output>, Resampler>;

template <typename State, typename Output, typename Resampler = MultinomialResampler<State, AlwaysScheme>>
using ExtendedPF = ParticleFilter<State, EKFSampler<State, Output>, Resampler>;

}  // namespace epf

#endif