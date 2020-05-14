# Reparameterizations of `GuidProp`
****************
In an MCMC setting we may wish to change the values of some parameters of `GuidProp`. `GuidProp` is defined as an immutable struct, so in-place change will usually be impossible; however, we can use a function `clone` to create a new instance of `GuidProp` with new parameter values and the remaining containers simply kept unchanged from a previous `GuidProp` instance

```@docs
GuidedProposals.clone
```

The first clone method is not working yet (but it's also not of much use).
The second method seems a bit convoluted, but it is all for optimizing performance for the MCMC setting. To understand how to use it consider the following example:

## Example
----------
```julia
# This `global vector` corresponds to all parameters of the MCMC chain
# (possibly many more than there in a single GuidProp)
ξ° = collect(1.0:10.0)

# These coordinates of the global vector are relevant for a `clone` call
sub_idx = [3, 5, 7]

# To each coordinate in `sub_idx` associate an index in `ξ`
invcoords = Dict(
	3 => 1,
	5 => 2,
	7 => 3,
)

# build `local vector` (corresponding to a local update of an MCMC chain)
ξ = ξ°[sub_idx] # [3.0, 5.0, 7.0]

# info on which parameters to update. For laws associate by name,
# for observations (not considered here) associate by position (instead of pname use local_idx)
θ°idx = [
	(global_idx=sub_idx[1], pname=:β),
	(global_idx=sub_idx[2], pname=:γ),
	(global_idx=sub_idx[3], pname=:α),
]

# clone
P_cloned = GP.clone(P, ξ, invcoords, θ°idx, [])
```
Now we have:
```julia
julia> DD.var_parameters(P_cloned.P_target)
Dict{Symbol,Float64} with 6 entries:
  :α  => 7.0
  :γ  => 5.0
  :δ  => 1.0
  :β  => 3.0
  :σ2 => 0.1
  :σ1 => 0.1

julia> DD.var_parameters(P_cloned.P_aux) == DD.var_parameters(P_cloned.P_target)
true
```

## Re-computing the guiding term
-----------------
Sometimes changing parameter values is all that needs to be done; however, it is not always the case. Sometimes changing parameter values implies that the terms used for computation of the guiding term also need to be updated. To see ...
