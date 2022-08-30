# EPF - Extendable Particle Filter

Particle filter has been studied since 20th century. Various improvements are proposed to solve the challenges encountered when applying the filter to real world problem. The main purpose of EPF is to try to implement particle filter in a composable manner, such that it is trivial to change a certain part of particle filter to suit one's need, from adding additional measurement model, to adopting different strategy for resampling step.

### Particle Filter

#### Terms

### Customization point (?)

#### 1. Particle Concept

Particle filter represents a distribution by a set of particles sampled from this distribution. In recursive state estimation, each particle is a concrete instantiation of the state at time *t*, i.e., a hypothesis as to what the true world state may be at time *t*.

The Particle Concept describes the requirement for a particle type. The requirements are checked by the concrete implementation of `epf::ParticleFilter` (e.g. `epf::AMCL2D`). Refer to each implementation for actual requirement.

#### 2. Particle Filter


### TODO