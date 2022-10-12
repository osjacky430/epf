#ifndef COMMON_PF_HPP_
#define COMMON_PF_HPP_

#include "component/importance_sampler/ukf.hpp"
#include "component/resampler/multinomial.hpp"
#include "core/particle_filter.hpp"

namespace epf {

template <typename State, typename Output, typename Resampler = MultinomialResample<State>>
using UPF = ParticleFilter<State, UKFSampler<State, Output>, Resampler>;

}  // namespace epf

#endif