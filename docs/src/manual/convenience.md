# Utility functions
We implemented some utility functions for computing the log-likelihoods, sampling the paths, accessing various elements related to guiding terms (and probably there will be more functions implemented in the near future).

## Computing the log-likelihoods
Probably the most important functions are `loglikhd` and `loglikhd_obs`. The former computes the logarithm of a Girsanov transform between the `target` and `proposal` laws evaluated at a sampled path:
```julia
ll° = loglikhd(XX[1], P[1])
```
The latter evaluates the logarithm of the auxiliary `ρ(x0,xT)` function (i.e. the contribution due to the end-points that is not canceled in importance/mcmc sampling schemes).
```julia
ll° = loglikhd_obs(P[1], x0)
```
