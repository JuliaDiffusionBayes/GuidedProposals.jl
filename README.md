<h1 align="center">
  <br>
  <a href="https://juliadiffusionbayes.github.io/GuidedProposals.jl/dev/"><img src="https://raw.githubusercontent.com/JuliaDiffusionBayes/GuidedProposals.jl/master/docs/src/assets/logo.png" alt="GuidedProposals.jl" width="200"></a>
  <br>
  GuidedProposals.jl
  <br>
</h1>

> Defining and sampling conditioned diffusion processes. A member of the suite of packages from [JuliaDiffusionBayes](https://github.com/JuliaDiffusionBayes).

<p align="center">
  <a href="https://JuliaDiffusionBayes.github.io/GuidedProposals.jl/stable">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg"
         alt="Stable">
  </a>
  <a href="https://JuliaDiffusionBayes.github.io/GuidedProposals.jl/dev"><img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Dev"></a>
  <a href="https://travis-ci.com/JuliaDiffusionBayes/GuidedProposals.jl">
      <img src="https://travis-ci.com/JuliaDiffusionBayes/GuidedProposals.jl.svg?branch=master" alt="Build Status">
  </a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#related">Related</a> •
  <a href="#license">License</a>
</p>


## Key features

Implementation of the [Backward Filtering&ndash;Forward Guiding *(BFFG)*](https://arxiv.org/pdf/1712.03807.pdf) algorithm. It is a generic computational framework for working with [Guided Proposals](https://projecteuclid.org/euclid.bj/1494316837) introduced by M Schauer, F van der Meulen and H van Zanten [[arXiv](https://arxiv.org/pdf/1311.3606.pdf)].

The main object introduced by this package is a struct `GuidProp` and it allows for sampling of guided proposals, computing log-likelihood functions of the sampled trajectories and embedding the samplers in smoothing or inference algorithms.

## Installation

The package is not yet registered. To install it, type in:
```julia
] add https://github.com/JuliaDiffusionBayes/GuidedProposals.jl
```

## How To Use

See [the documentation](https://juliadiffusionbayes.github.io/GuidedProposals.jl/dev/).

## Related

GuidedProposals.jl belongs to a suite of packages in [JuliaDiffusionBayes](https://github.com/JuliaDiffusionBayes), whose aim is to facilitate Bayesian inference for diffusion processes. Some other packages in this suite are as follows:
- [DiffusionDefinition.jl](https://github.com/JuliaDiffusionBayes/DiffusionDefinition.jl): define diffusion processes and sample from their laws
- [ObservationSchemes.jl](https://github.com/JuliaDiffusionBayes/ObservationSchemes.jl): a systematic way of encoding discrete-time observations for stochastic processes
- [ExtensibleMCMC.jl](https://github.com/JuliaDiffusionBayes/ExtensibleMCMC.jl): a modular implementation of the Markov chain Monte Carlo (MCMC) algorithms
- [DiffusionMCMCTools.jl](https://github.com/JuliaDiffusionBayes/DiffusionMCMCTools.jl): utility methods that facilitate easier coding solutions for smoothing and inference algorithms for diffusions
- [DiffusionMCMC.jl](https://github.com/JuliaDiffusionBayes/DiffusionMCMC.jl): Markov chain Monte Carlo (MCMC) algorithms for doing inference for diffusion processes

## License

MIT
