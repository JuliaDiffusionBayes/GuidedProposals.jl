# Guided proposals with multiple observations
In practice, we want to sample from some target diffusion law
```math
\dd X_t = b(t,X_t)\dd t + \sigma(t,X_t) \dd W_t,\quad t\in[0,T],\quad X_0\sim p_0,
```
conditionally on multiple, partial observations of $X$:
```math
V_{t_i}:=\left\{ L_iX_{t_i}+\eta_i;i=1,\dots,N \right\},\quad L_i\in\RR^{m_i\times d},\quad \eta_i\sim N(\mu_i,\Sigma_i).
```
This can be done by stacking together `GuidProp`, each defined on its own interval $[t_{i-1},t_{i}]$ and for its own terminal observation $V_{t_i}$. To properly initialize the guiding term we must defined the sequence of `GuidProp` starting from the last interval $[t_{N-1},t_{N}]$ and proceed moving backwards: $[t_{N-2},t_{N-1}],\dots,[0,t_{1}]$, each time passing a `GuidProp` from the subsequent interval $[t_{i},t_{i+1}]$ to the `GuidProp` that is being defined on $[t_{i-1},t_{i}]$.

Let's look at an example with three observations:
```julia
observs = [
    LinearGsnObs(
        1.0, (@SVector [1.0, 2.0]);
        Σ = SDiagonal(1.0, 1.0)
    ),
    LinearGsnObs(
        2.0, (@SVector [1.5]);
        L = (@SMatrix [1.0 0.0]), Σ = SDiagonal(1.0)
    )
    LinearGsnObs(
        3.0, (@SVector [2.0, 3.0]);
        Σ = SDiagonal(1.0, 1.0)
    )
]
```
We should define three `GuidProp` structs as follows:
```julia
dt = 0.01
tts = [
    0.0:dt:(observs[1].t),
    (observs[1].t):dt:(observs[2].t),
    (observs[2].t):dt:(observs[3].t),
]
P_intv3 = GuidProp(tts[3], P_target, LotkaVolterraAux, observs[3])
P_intv2 = GuidProp(tts[2], P_target, LotkaVolterraAux, observs[2]; next_guid_prop=P_intv3)
P_intv1 = GuidProp(tts[1], P_target, LotkaVolterraAux, observs[1]; next_guid_prop=P_intv2)
```
That's it, now a vector
```julia
P = [P_intv1, P_intv2, P_intv3]
```
Defines everything we need to sample guided proposals with multiple conditionings.
## Simpler syntax
Performing steps above is so common that we introduced an additional utility function that does the steps above for the user (and comes with some additional flexibility, more about it in ...). To define multiple guided proposals for multiple observations we first package the observations in a format of a `recording` from [DiffObservScheme.jl](https://github.com/JuliaDiffusionBayes/DiffObservScheme.jl), so for the example above our `recording` would become:
```julia
recording = (
    P = P_target,
    obs = observs,
    t0 = 0.0,
    x0_prior = undef # normally, we would provide a prior, however for the steps
    # below it is not needed
)
```
Then, to define guided proposals for a given `recording` we can call
```julia
P = standard_build_guid_prop(LotkaVolterraAux, recording, tts)
```
That's it. `P` is now equivalent to a vector `[P_intv1, P_intv2, P_intv3]`.
