# Computing path functionals
****************************
This package uses the Euler-Maruyama solvers implemented in [DiffusionDefinition.jl](https://juliadiffusionbayes.github.io/DiffusionDefinition.jl/dev/) to sample diffusion trajectories. Consequently, in the same way that in [the latter package it has been possible to compute path functionals whilst sampling](https://juliadiffusionbayes.github.io/DiffusionDefinition.jl/dev/manual/functionals_of_paths/), it is possible to do so for the conditioned trajectories of guided proposals sampled with this package as well.

## Sampling over a single interval
-----------------------
To compute a functional when sampling over a single interval simply pass a named argument `f=...` with a function you wish to evaluate. The function that you pass must have the following methods defined for it:
```julia
# called at the very start of solve!
f_accum = foo(P, y)
# called at the beginning of every iteration of the Euler-Maruyama scheme
f_accum = foo(f_accum, P, y, t, dt, dW, i)
# called at the very end, just before return statement
f_accum = foo(f_accum, P, y, Val(:final))
```
For in-place computations these three functions must have a slightly different form
```julia
# called at the very start of solve!
f_accum = foo(buffer, P, y) # the buffer needs to accommodate the needs of function f
# called at the beginning of every iteration of the Euler-Maruyama scheme
foo(buffer, P, y, tt[i-1], dt, i-1)
# called at the very end, just before return statement
foo(buffer, P, y, _FINAL)
```
You may pass it to `rand` or `rand!` so long as `Val(:ll)` is not passed (i.e. so long as you are not calling optimized sampler that already computes the log-likelihood for you).

## Sampling over multiple intervals
-----------------------
To compute functionals when sampling over multiple intervals, instead of passing a single method as a named argument `f=...` pass a list of methods, say `f=[foo₁,foo₂,foo₃]`, one for each interval. Additionally, pass an extra container to a named argument `f_out`. The results of method evaluation on each interval are going to be saved there.

# Computing gradients of path functionals
*************
Just as it was described in [DiffusionDefinition.jl](https://juliadiffusionbayes.github.io/DiffusionDefinition.jl/dev/), it is possible to compute gradients while sampling.
!!! warning "TODO"
    Complete the description.
