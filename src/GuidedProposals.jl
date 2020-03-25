module GuidedProposals

using DiffEqCallbacks, OrdinaryDiffEq, LinearAlgebra
using StaticArrays

#these need to be defined:
B() = nothing
β() = nothing
σ() = nothing
a() = nothing
B!() = nothing
β!() = nothing
σ!() = nothing
a!() = nothing
dimension() = nothing

include("utility_functions.jl")
include("ode_states_accessors.jl")
include("ode_solver_hfc.jl")
include("ode_solver_mlmu.jl")
include("ode_solver_pnu.jl")
include("guided_proposals.jl")

export GuidProp, H, F, c

end # module
