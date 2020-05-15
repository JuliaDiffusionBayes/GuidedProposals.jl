# [Installation](@id get_started)
The package is not registered yet. To install it write:
```julia
] add https://github.com/JuliaDiffusionBayes/GuidedProposals.jl
```

!!! note
    The package depends on [DiffusionDefinition.jl](https://github.com/JuliaDiffusionBayes/DiffusionDefinition.jl) and [ObservationSchemes.jl](https://github.com/JuliaDiffusionBayes/ObservationSchemes.jl), neither of which is registered yet. Install them in the same way as [GuidedProposals.jl](https://github.com/JuliaDiffusionBayes/GuidedProposals.jl).

## Define the law of a guided proposal
-----------------
The main object defined by this package is `GuidProp`. It allows for a definition of a [guided proposal](https://projecteuclid.org/euclid.bj/1494316837) [[arXiv](https://arxiv.org/abs/1311.3606)] with some target and auxiliary diffusion laws. To define it, use [DiffusionDefinition.jl](https://github.com/JuliaDiffusionBayes/DiffusionDefinition.jl) to define the unconditioned laws, [ObservationSchemes.jl](https://github.com/JuliaDiffusionBayes/ObservationSchemes.jl) to define the observation, and then, construct a `GuidProp`
```julia
using GuidedProposals, DiffusionDefinition, ObservationSchemes
const GP = GuidedProposals
const DD = DiffusionDefinition
const OBS = ObservationSchemes

using StaticArrays, LinearAlgebra

@load_diffusion LotkaVolterra
@load_diffusion LotkaVolterraAux

# define target law
θ = [2.0/3.0, 4.0/3.0, 1.0, 1.0, 0.1, 0.1]
P_target = LotkaVolterra(θ...)

# define the observation and a time-grid
T, vT = 3.0, @SVector [0.5, 0.8]
tt, y1 = 0.0:0.001:T, @SVector [2.0, 0.25]
obs = LinearGsnObs(T, vT; Σ=1e-4*SDiagonal{2,Float64}(I))

# define a guided proposal
P = GuidProp(tt, P_target, LotkaVolterraAux, obs)
```

## Define the laws of multiple guided proposals
------------------
If there are more than one (non-full) observations, then you should construct one `GuidProp` for each inter-observation interval. This is done automatically with `build_guid_prop`:
```julia
# multiple obsevations
observs = load_data(
    ObsScheme(
        LinearGsnObs(
            0.0, zero(SVector{2,Float64});
            Σ = 1e-4*SDiagonal(1.0, 1.0)
        )
    ),
    [1.0, 2.0, 3.0],
    [[2.2, 0.7], [0.9, 1.0], [0.5, 0.8]]
)

# packaged in a format of a `recording` from ObservationSchemes.jl
recording = (
    P = P_target,
    obs = observs,
    t0 = 0.0,
    x0_prior = undef
)
tts = OBS.setup_time_grids(recording, 0.001)

# create a guided proposal for multiple observations
PP = build_guid_prop(LotkaVolterraAux, recording, tts)
```

## Sample guided proposals
-------------
Sampling is done with a function `rand`:
```julia
X, W, Wnr = rand(P, y1)
```
that initializes all containers, or with `rand!` if you initialize the containers yourself:
```julia
X, W = trajectory(P)
rand!(P, X, W, y1; Wnr=Wnr)
```
The functions above work for multiple `GuidProp` as well (that correspond to multiple observations):
```julia
XX, WW, Wnr = rand(PP, y1)
# OR
XX, WW = trajectory(PP)
rand!(PP, XX, WW, y1)
```

## Compute log-likelihoods
-----------
Computation of the log-likelihood may happen after the path has been sampled:
```julia
ll_path_contrib = loglikhd(P, X)
ll_obs_contrib = loglikhd_obs(P, y1)
ll = ll_path_contrib + ll_obs_contrib
# OR when passing multiple observations:
ll = loglikhd(PP, XX)
```
or at the time of sampling
```julia
_, ll_path_contrib = rand!(P, X, W, Val(:ll), y1; Wnr=Wnr)
ll_obs_contrib = loglikhd_obs(P, y1)
ll = ll_path_contrib + ll_obs_contrib
# OR when passing multiple observations:
_, ll = rand!(PP, XX, WW, Val(:ll), y1; Wnr=Wnr)
```

## Compute path functionals
------
To evaluate functionals while sampling simply pass the method as a named argument `f=...`.
```julia
X, W, Wnr, f_out = rand(P, y1; f=foo)
```

## Compute gradients of log-likelihood or path functionals
----
!!! warning "TODO"
    to be written-up
