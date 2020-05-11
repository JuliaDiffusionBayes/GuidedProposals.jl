# Backward filtering-forward guiding algorithm
This package implements a computational framework from [this paper](https://arxiv.org/abs/1712.03807) for computing various terms of guided proposals. In particular, we provide high-level routines that correspond to the two main sweeps of the main algorithm:
- `backward_filter!` &
- `forward_guide!`.
Both of them operate on an already initialized vector of `GuidProp` that corresponds to a `recording`.
## Forward guide
The `forward_guide!` function is used to sample a trajectory of multiple guided proposals stacked together, targeting some conditional diffusion law, that is conditioned on multiple observations.

Let's look at the example from the previous section, were we defined guided proposals
for a recording with three observations. We first formalize the definition of a recording by adding a prior over the starting point and then define the guided proposals again:
```julia
recording = (
    P = P_target,
    obs = observs,
    t0 = 0.0,
    x0_prior = KnownStartingPt((@SVector [0.5 0.5])),
)
P = standard_build_guid_prop(LotkaVolterraAux, recording, tts)
```
Because these functions have been created with the thought in mind of being able to call them repeatedly over and over again in the most efficient way we should now define a couple of containers that are needed to store intermediate states of various simulated objects. These are:
- A container for sampled driving Brownian motion
- A container for sampled trajectory of a proposal path
- A vector with flags for dimensions/shape/data type of Brownian motions
- A vector of guided proposals
Additionally we must provide a starting point to start sampling from, and finally, optionally, we can provide another container with a realization of a Brownian motion and a vector of hyper-parameters (each taking values $\in [0,1]$) for doing the preconditioned Crank-Nicolson scheme. For simplicity let's leave the last to at default values (see the source code for more info). Then, we have the following:
```julia
WW° = map(tt->trajectory(tt, DD.ℝ{2}), tts)
XX° = deepcopy(WW°)
Wnr = [Wiener() for _=1:3]
x0 = rand(recording.x0_prior)
```
We can now sample guided proposals with `forward_guide!`. The function will output a flag for whether the sampling has been successful (it can fail if the sampled path hits a boundary of the state space on which the process is defined) as well as the log-likelihood (due to sampled path only! in particular the contribution due to end-points is not computed and need to be computed explicitly by calling `loglikhd_obs` (though the two might be joined in the future)) that is computed alongside the sample. The sampled path is going to be saved to `XX°`.
```julia
success, ll° = forward_guide!(WW°, XX°, Wnr, P, x0)
```
The sampled path is stored in a vector of trajectories, so we can use a generic plotting functionality suitable for trajectories from  [DiffusionVis.jl](https://github.com/JuliaDiffusionBayes/DiffusionVis.jl) (NOTE: not implemented).

As a side note, `forward_guide!` can also be applied directly to an interval so that sampling is done only for a single `GuidProp`, for instance, we could call:
```julia
success_single, ll_single° = forward_guide!(WW°[1], XX°[1], Wnr[1], P[1], x0)
```

## Backward filter
A very important functionality of `GuidProp` is the ability to change the parameters of `GuidProp` without having to-reallocate almost any memory. This can be done by calling a `GuidProp` constructor that is specialized in doing such reassignment of parameters:
```julia
θ° = ... # some new parameterization of the diffusion law
η° = ... # some new parameterization of the observation
P1_new = GuidProp(P[1], θ°, η°)
```
Then, internally the parameters in the diffusion laws and observations are changed, but nothing is done about recomputing the guiding terms. Sometimes nothing has to be done, because the parameters have no influence on the computation of the guiding term (in which case we save some resources by avoiding expensive calls to the ODE solvers), other times, the ODE systems need to be solved anew. To re-compute the guiding term for the new parameters all we need to do is call `recompute_guiding_term!`:
```julia
recompute_guiding_term!(P1_new)
```
**NOTE** that the above would not only change the guiding term of `P1_new`, but also of `P[1]`, because the constructor `GuidProp(P[1], θ°, η°)` simply re-uses the guiding term solving machinery of `P[1]`.

Now, in practice we will be working with `recording`s and not a single observation, so we will instead have:
```julia
θ° = ... # some new parameterization of the diffusion law
ηs° = [η1°,...] # some new parameterization of the observations
P_new = map(i->GuidProp(P[i], θ°, ηs°[i]), 1:length(P))
```
and then instead of calling `recompute_guiding_term!` we should call a `backward_filter!` (which is just a wrapper function around multiple `recompute_guiding_term!`s that work on multiple `GuidProp` stacked together).
```julia
backward_filter!(P_new)
```
That's it. Now, `P_new` has new parameters, has its guiding term re-computed to be compatible with those parameters and can be used for sampling.
