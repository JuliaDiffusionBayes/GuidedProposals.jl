# Additional options for `GuidProp` definition
In the section on defining `GuidProp` we were deliberately brief and did not go into full detail regarding passing parameters that can define `GuidProp`. Here, we go into detail.

Additional set of options passed to `GuidProp` are specified in the field `solver_choice`, which by default is set to:
```julia
solver_choice=(
      solver=Tsit5(),
      ode_type=:HFc,
      convert_to_HFc=false,
      mode=:outofplace,
      gradients=false,
      eltype=Float64,
)
```
The constructor expects it to be a NamedTuple with the respective fields (but it is robust enough to fill-in any missing fields with defaults). The meaning of the fields is as follows:
- `solver` is passed to `DifferentialEquations.jl` to pick an algorithm for solving ODEs that define the guiding term. More about the ODE systems is written in the following section.
- `ode_type` is used to pick between three choices of ODE systems to use: `H`, `F`, `c` system, `M`, `L`, `μ` system and `P`, `ν` (and `c`, but `c` needs to be added to names). They have the labels: `:HFc`, `MLμ`, `Pν` respectively, which are not case sensitive and currently only `HFc` is implemented)
- `convert_to_HFc` is used only when `:MLμ` has been chosen to be a solver of ODEs. In that scenario, if `convert_to_HFc` is set to `true`, then the terms `M`, `L`, `μ` that the ODE systems solve for will be used to compute the corresponding `H`, `F`, `c` terms (as opposed to using `:HFc` solver to solve for them)
- `mode` is an important flag (currently only `:outofplace` is fully supported) and it is used to tell `GuidProp` what type of computations are being performed: out-of-place `:outofplace`, which are based on `SVector`s from (StaticArrays.jl)[https://github.com/JuliaArrays/StaticArrays.jl], in-place `:inplace`, which are based on `Vector`s or `:gpu`, which are based on `cuArray`s.
- `gradients` is another important flag for telling `GuidProp` whether gradients with respect to something need to be computed.
- `eltype` ignore this for a moment, we need to figure some things out with this...
