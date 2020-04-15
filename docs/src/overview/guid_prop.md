# Defining guided proposals
The main object used to define guided proposals is a struct `GuidProp`. Its main role is to facilitate sampling of diffusion paths from some `target` diffusion law by:
- drawing from a `proposal` diffusion law &
- computing an importance sampling weight
The two can be used in, say, an importance sampler or mcmc sampler to draw sample-paths from the `target` law.

The following information (listed in an abstract form) is required to define this object:
- A `target` diffusion law
- An `auxiliary` diffusion law (which is used together with the `target` law to internally define the `proposal` law)
- observation of the target process that sampling is conditioned-on
- time-grid on which the path will be sampled on
Mathematically speaking, guided proposals cannot be defined without the first three elements above; however, the fourth element is superfluous. However, for practical purposes we ask for the time-grid on which the process can be sampled on to be fixed in advance.

## Defining the four elements that characterize `GuidProp`
We first need to define the four elements from the list above. To this end, we should make use of two other packages from the [JuliaDiffusionBayes](https://github.com/JuliaDiffusionBayes/) suite: [DiffusionDefinition.jl](https://github.com/JuliaDiffusionBayes/DiffusionDefinition.jl) and [DiffObservScheme.jl](https://github.com/JuliaDiffusionBayes/DiffObservScheme.jl). We can use the former package to define the `target` and `auxiliary` diffusion. We can either define them using the macro `@define_diffusion` or if we can, simply load in the pre-defined processes. Let's do the latter for simplicity:
```julia
using DiffusionDefinition
const DD = DiffusionDefinition
@load_diffusion LotkaVolterra # for constructing the target law
@load_diffusion LotkaVolterraAux # for the auxiliary law

# let's construct the target law
θ = [2.0/3.0, 4.0/3.0, 1.0, 1.0, 0.2, 0.2]
P_target = LotkaVolterra(θ...)
```
**IMPORANT** An important note about defining the auxiliary law using `@define_diffusion` macro. Currently, the following fields from `:auxiliary_info` must be defined: `:t0`, `:T` and `:vT`. Additionally, the field `:xT` will be defined automatically and (if need be) auto-initialized (but can also be declared explicitly). These fields are expected by the `GuidProp` constructor to exist. No other field from `:auxiliary_info` (apart from explicitly declaring `:xT`) can be used (i.e. fields `:v0` and `:x0`). This last restriction will probably be relaxed in the near future.

Unlike the target, the auxiliary law need not be initialized. This is because once `GuidProp` is given an instantiated `P_target`, an observation `obs` and a template for constructing the auxiliary law (in the example above, it is given by a definition of a struct `LotkaVolterraAux`) the constructor has all the necessary information to initialize `P_aux` itself. In this way we avoid asking the user to duplicate some actions (such as having to initialize `P_target` and `P_aux` with the same parameter `θ`).

We can now use [DiffObservScheme.jl](https://github.com/JuliaDiffusionBayes/DiffObservScheme.jl) to define an observation. Let's initialize an observation sampled according to a scheme:
$$
v=X+\eta,\quad \eta\sim N(0,I).
$$
```julia
using DiffObservScheme, StaticArrays
t, xₜ = 1.0, (@SVector [1.0, 2.0])
obs = LinearGsnObs(t, x_t; Σ = SDiagonal(1.0, 1.0))
```
We can now define a time-grid for the interval $[0,t]$ on which a guided proposal will be defined:
```julia
dt = 0.01
tt = 0.0:dt:t
```
say. We can now define guided proposals that can be used for sampling on the interval `[0,t]` from the target law `LotkaVolterra(θ...)` conditioned on `obs` in an importance sampling setting:
```julia
P = GuidProp(tt, P_target, LotkaVolterraAux, obs)
```
At initialization a sequence of computations is performed that derive a `guiding term` for `t`'s lying on a pre-specified time-grid `tt`, as well as some additional quantities need for computations of the `log-likelihoods`. Therefore `P` above is ready to be sampled from.

If at any point the parameters change, then the `guiding term` might need to be re-computed (in fact, this is the centerpiece of the `backward filtering` part of the `forward filtering-backward guiding` algorithm). We provide certain utility functions that facilitate these operations. See ... for more details.

#### Note on fixing the time-grid
Conceptually, guided proposals are defined as continuous-time processes, and thus, it should be possible to look-up the value of any sampled trajectory at any time $t\in[0,T]$. Nonetheless, for the purposes that this package was created fixing a time-grid `tt` at initialization of any `GuidProp` is helpful in reducing the computational cost of the algorithm. Currently, if sampling on finer grid is needed at any point, then `GuidProp` object needs to be redefined. This property is not set in stone however, as the computationally intensive routines rely on `DifferentialEquations.jl` and it is reasonably easy to introduce modifications that could leverage the flexibility provided by `DifferentialEquations.jl` to allow for refinements of `tt` post-initialization of `GuidProp`. Due to lack of use-cases this extension is not on any TODO lists.
