# GuidedProposals.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDiffusionBayes.github.io/GuidedProposals.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDiffusionBayes.github.io/GuidedProposals.jl/dev)
[![Build Status](https://travis-ci.com/JuliaDiffusionBayes/GuidedProposals.jl.svg?branch=master)](https://travis-ci.com/JuliaDiffusionBayes/GuidedProposals.jl)

## Scope and intended use

Implementation of the [Backward Filtering&ndash;Forward Guiding](https://arxiv.org/pdf/1712.03807.pdf) algorithm. It is a generic computational routine for implementing [Guided Proposals](https://projecteuclid.org/euclid.bj/1494316837) introduced by M Schauer, F van der Meulen and H van Zanten [[arXiv](https://arxiv.org/pdf/1311.3606.pdf)].

This package is an integral part of the [JuliaDiffusionBayes](https://github.com/JuliaDiffusionBayes) suite of packages designed to perform Bayesian inference for discretely observed diffusions. However, this package can be used outside of this ecosystem (for instance, to facilitate importance sampling on a path space with Guided Proposals) so long as it is accompanied by the two other packages: [DiffusionDefinition.jl](https://github.com/JuliaDiffusionBayes/DiffusionDefinition.jl) and [DiffObservScheme.jl](https://github.com/JuliaDiffusionBayes/DiffObservScheme).

## The main functionality
The main object introduced by this package is a `struct`: `GuidProp`. Each instance of `GuidProp` stores information about the target and the auxiliary diffusion laws on a given time-interval, observation at an end-point and additional "computational workhorse" object that performs heavy lifting in computing all the necessary terms for defining a proposal diffusion law.

To define a guided proposal we must choose a target and an auxiliary diffusion laws, we can make use of some pre-defined diffusions in the package [DiffusionDefinition.jl](https://github.com/JuliaDiffusionBayes/DiffusionDefinition.jl):
```julia
using DiffusionDefinition
const DD = DiffusionDefinition
@load_diffusion LotkaVolterra
@load_diffusion LotkaVolterraAux
θ = [2.0/3.0, 4.0/3.0, 1.0, 1.0, 0.2, 0.2]
P_target = LotkaVolterra(θ...)
```
**IMPORTANT** the auxiliary diffusion law must have `:auxiliary_info` fields `:t0`, `:T`, `:vT`, as these fields are expected to be populated in the constructor of `GuidProp`. This requirement will be lifted soon, but probably it will not be extended any time soon to optionally include `:v0` and `:x0` any time soon, as these two are not used very often and are a bit awkward to fetch.

The auxiliary diffusion does not need to be initialized. Additionally, we must define an observation and to this end we can use [DiffObservScheme.jl](https://github.com/JuliaDiffusionBayes/DiffObservScheme):
```julia
using DiffObservScheme, StaticArrays
t, x_t = 1.0, (@SVector [1.0, 2.0])
obs = LinearGsnObs(t, x_t; Σ = SDiagonal(1.0, 1.0))
```
`GuidProp` can now be defined simply as:
```julia
using GuidedProposals
tt = 0.0:0.01:t
P = GuidProp(tt, P_target, LotkaVolterraAux, obs)
```
Where `tt` is a time grid on which the terms needed for proposal law are revealed. That's it!

## More than one observation
If we have more than one observation, then we should stack `GuidProp` together:
```julia
observs = [
    LinearGsnObs(1.0, (@SVector [1.0, 2.0]); Σ = SDiagonal(1.0, 1.0)),
    LinearGsnObs(2.0, (@SVector [2.0, 3.0]); Σ = SDiagonal(1.0, 1.0))
]
tts = [0.0:0.01:1.0, 1.0:0.01:2.0]
P_intv2 = GuidProp(tts[2], P_target, LotkaVolterraAux, obs[2])
P_intv1 = GuidProp(tts[1], P_target, LotkaVolterraAux, obs[1]; next_guid_prop=P_intv2)
```
where notice that we start from defining the guided proposal `P_intv2` on the second interval `[1,2]` first and we then feed the outputted object into a constructor of a guided proposal on the inteval `[0,1]`.

## Additional functionality

Sample with `forward_guide!`, change parameters with `GuidProp(P::GuidProp, θ°, η°)` constructor, recompute ODEs with `backward_filter!` (or `recompute_guiding_term!` if there is only one `GuidProp`), compute log-likelihood with `log_likhd` and `log_likhd_obs`. See [documentation](https://JuliaDiffusionBayes.github.io/GuidedProposals.jl/dev) for more details (it is empty now...).
