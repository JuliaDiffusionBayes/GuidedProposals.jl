# Backward filtering-forward guiding algorithm
**************
Backward filtering-forward guiding algorithm is a computational framework for sampling conditioned diffusions, it has been described in detail in [this paper](https://arxiv.org/abs/1712.03807) and it is what happens behind the scenes of [GuidedProposals.jl](https://juliadiffusionbayes.github.io/GuidedProposals.jl/dev/).

Briefly, it comprises of two passes, that—if repeated many times—may result in a smoothing or inference algorithms. The two steps are:
- backward filtering &
- forward guiding.

## Backward filtering
-------
Backward filtering is a technical term for propagating the information about the observations (that lie in the future) back into the parts of the interval that precede them (and on which the diffusion is defined). It is simply the process of computing the guiding term. It is done implicitly every time `GuidProp` is instantiated. Additionally, after parameters change it may also be done explicitly by calling `recompute_guiding_term!` or its alias
```@docs
backward_filter!
```
see the [section on reparameterizations](@ref recompute_guiding_term) for more details.

## Forward guiding
----
Forward guiding is a technical term for sampling guiding proposals. It is encapsulated by various `rand`, `rand!` and `solve_and_ll!` routines. You may also call
```@docs
GuidedProposals.forward_guide!
```
See sections [Defining guided proposals](@ref manual_start) and [Guided proposals with multiple observations](@ref multi_obs_gp) for more details about sampling.

!!! tip
    Check-out the how-to guides and tutorials on smoothing and inference to see the *BFFG* algorithm in action.
