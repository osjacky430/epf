#ifndef COMMON_PF_HPP_
#define COMMON_PF_HPP_

#include "epf/component/importance_sampler/ekf.hpp"
#include "epf/component/importance_sampler/ukf.hpp"
#include "epf/component/resampler/multinomial.hpp"
#include "epf/core/particle_filter.hpp"

namespace epf {

template <typename State, typename Output, typename Resampler = MultinomialResample<State>>
using UnscentedPF = ParticleFilter<State, UKFSampler<State, Output>, Resampler>;

template <typename State, typename Output, typename Resampler = MultinomialResample<State>>
using ExtendedPF = ParticleFilter<State, EKFSampler<State, Output>, Resampler>;

}  // namespace epf

#endif