# GuidedProposals.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDiffusionBayes.github.io/GuidedProposals.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDiffusionBayes.github.io/GuidedProposals.jl/dev)
[![Build Status](https://travis-ci.com/JuliaDiffusionBayes/GuidedProposals.jl.svg?branch=master)](https://travis-ci.com/JuliaDiffusionBayes/GuidedProposals.jl)

## Scope and intended use

Implementation of the [Backward Filtering&ndash;Forward Guiding *(BFFG)*](https://arxiv.org/pdf/1712.03807.pdf) algorithm. It is a generic computational framework for working with [Guided Proposals](https://projecteuclid.org/euclid.bj/1494316837) introduced by M Schauer, F van der Meulen and H van Zanten [[arXiv](https://arxiv.org/pdf/1311.3606.pdf)].

This package is an integral part of the suite [JuliaDiffusionBayes](https://github.com/JuliaDiffusionBayes), designed to perform Bayesian inference for discretely observed diffusion processes. It may also be used outside of this ecosystem (for instance, to facilitate importance sampling on a path space via Guided Proposals) so long as it is accompanied by two other packages: [DiffusionDefinition.jl](https://github.com/JuliaDiffusionBayes/DiffusionDefinition.jl) and [DiffObservScheme.jl](https://github.com/JuliaDiffusionBayes/DiffObservScheme).

## The main functionality

The main object introduced by this package is a struct `GuidProp` and it allows for sampling of guided proposals, computing log-likelihood functions of the sampled trajectories and embedding the samplers in smoothing or inference algorithms.

For an overview, detailed introduction to, tutorials and how-to guides for this package see the [documentation](https://JuliaDiffusionBayes.github.io/GuidedProposals.jl/dev).
