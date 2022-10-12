# EPF - Extendable Particle Filter

Particle filter has been studied since 20th century. Various improvements are proposed to solve the challenges encountered when applying the filter to real world problem. The main purpose of EPF is to try to implement particle filter in a composable manner, such that it is trivial to change a certain part of particle filter to suit one's need, from adding additional measurement model, to adopting different strategy for resampling step.

# Table of content
- [EPF - Extendable Particle Filter](#epf---extendable-particle-filter)
- [Table of content](#table-of-content)
- [Terminologies](#terminologies)
  - [State-Space Model](#state-space-model)
  - [Particle Filter](#particle-filter)
    - [Importance Sampling](#importance-sampling)
    - [Resampling](#resampling)
  - [Particle (State) Concept](#particle-state-concept)
- [TODO](#todo)
- [Reference](#reference)

# Terminologies

There are countless of literatures, papers talking about particle filter. Terms may vary according to disciplines. Therefore, it is common to spend a stupid amount of time, only to find that we are referring to different things, despite having same notation, which is extremely frustrating. To avoid that, we start by defining some terms that we will use in the following sections. (TODO: finish it)

## State-Space Model
Consider a general, non-linear state-space model:

$\textbf{x}_{t} = f\,(\textbf{x}_{t-1}, \textbf{u}_{t}, \textbf{v}_{t-1})$  
$\textbf{y}_{t} = h\,(\textbf{x}_{t}, \textbf{u}_{t}, \textbf{n}_{t})$  

where:

$ \textbf{x}_{t} $ : state of the system at time t, with covariance $ \textbf{P}_{\textbf{x}\textbf{x}} $  
$ \textbf{y}_{t} $ : observation, with covariance $ \textbf{P}_{\textbf{y}\textbf{y}} $  
$ \textbf{u}_{t} $ : input  
$ \textbf{v}_{t} $ : process noise, with noise covariance $ \textbf{Q}_{t} $  
$ \textbf{n}_{t} $ : observation noise with noise covariance $ \textbf{R}_{t} $

The general state-space model can be broken down into a state transition and state measuremment model, these two probabilities are calculated during `(unify this term!)` in `epf::ProcesssModel` and `epf::MeasurementModel`, respectively:  

- $ p\,(\textbf{x}_t\,|\,\textbf{x}_{t-1}\,,\,\textbf{u}_{t}) $
- $ p\,(\textbf{y}_{t}\,|\,\textbf{x}_{t}) $

The estimation of the state $ \textbf{x}_{t} $ is expressed as conditional probability, sometimes referred to as belief, $ p\,(\textbf{x}_{t}\,|\,\textbf{ y}_{1:t},\,\textbf{u}_{1:t}) $. Assume that the state $ \textbf{x}_{t} $ is *complete* (or is a Markov process), meaning that the past states, measurements, or controls provide no additional information that would help us predict the future more accurately, $ p\,(\textbf{x}_{t}\,|\,\textbf{ y}_{1:t}\,,\,\textbf{u}_{1:t}) = p\,(\textbf{x}_t\,|\,\textbf{y}_t\,,\,\textbf{u}_t) $, this posterior probability is the target particle filter aims to approximate. (see (Intelligent Robotics and Autonomous Agents) Sebastian Thrun_ Wolfram Burgard_ Dieter Fox - Probabilistic Robotics-Mit Press (2005))

Particle filter is widely used in engineering, such as robotics, signal processing, etc. (reference). Therefore, EPF also implemented some often-used process models and measurement models, see `component\process_model\` and `component\measurement_model\` for more detail (p.s. more implementations are welcome!).

## Particle Filter

Particle filter is a nonparametric implementation of the Bayes filter, it approximates the distribution by a set of random state drawn from it. There are multiple ways to calculate the approximations, the default implementation of EPF (see `epf::ParticleFilter::sample`) follows the structure of the most popular one, the bootstrap filter, also known as Sequential Importance Sampling with Resampling (SISR). The SISR filter can be broken down into two steps, i.e. Importance Sampling and Resampling. (see Particle Filters and Bayesian Inference in Financial Econometrics, THE UNSCENTED PARTICLE FILTER)

```mermaid
flowchart LR
  subgraph init["t = 0"]
    direction LR
    Init["Initialization"]
  end
  subgraph main_algo["t = 1,2,3,..."]
    direction LR
    IS["Importance Sampling"] --> R["Resampling"]
  end
  init --> main_algo --> Output
```

### Importance Sampling

As mentioned previously, our goal is to approximate the state at time *t*. Unfortunately, it is often impossible to sample directly from the posterior distribution. However, we can overcome this difficulty by sampling from a known, easy-to-sample, **proposal distribution**, $ q\,(\,\textbf{x}_{0:t}\,|\,\,\textbf{y}_{1:t}\,) $ , with importance weight $ w_t \propto \frac{p (\textbf{y}_{t}|\textbf{x}_{t})\,\cdot\,p(\textbf{x}_t|\textbf{x}_{t-1})}{q(\textbf{x}_t|\textbf{x}_{0:t-1},\textbf{y}_{1:t})} $.

The implementation of the proposal distribution is under directory `component\importance_sampler`, the default importance sampling strategy is $ q(\textbf{x}_t|\textbf{x}_{0:t-1},\textbf{y}_{1:t}) = p(\textbf{x}_t|\textbf{x}_{t-1}) $, i.e., importance weight $ w_t \propto p (\textbf{y}_{t}|\textbf{x}_{t}) $, see `component\importance_sampler\default.hpp`

### Resampling

Importance sampling along can't do much since it will degenerate with time (see THE UNSCENTED PARTICLE FILTER), luckily, with the inclusion of resampling stage, particle filter is finally a practical filter. The basic idea is that, resampling stage draws particle from importance sampling according to their importance weight, i.e., particles with higher weights will stay, whereas lower one will be discarded (kinda like the Darwinian idea of survival of the fittest). Different resampling strategies may affect the "performace" of the particle filter (though the author in THE UNSCENTED PARTICLE FILTER found that those do not affect the "performace" of it). Nevertheless, EPF implements different resampling strategy under directory `component\resampler`, the default strategy uses multinomial sampling. (see reference)

## Particle (State) Concept

The Particle Concept describes the requirement for a particle type. The requirements are checked by each component, e.g. `epf::Differential`, `epf::ParticleFilter`. Refer to each component for actual requirement.

# TODO
1. support multi-thread or coroutine
2. make it harder to misuse (this is a bit vague, basically, I need to come back to some classes since some of them should be treated specially, but I haven't think about it thoroughly)
3. make it easy to install
4. adapt to ros (this can only be done after #3)
5. more example

# Reference
1. particle-filter-tutorial
2. papers.. 