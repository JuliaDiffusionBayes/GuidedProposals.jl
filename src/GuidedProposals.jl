module GuidedProposals

using DiffEqCallbacks, OrdinaryDiffEq, LinearAlgebra
using StaticArrays
using DiffusionDefinition
const DD = DiffusionDefinition

#=these need to be defined:
function B end
function β end
function σ end
function a end
function B! end
function β! end
function σ! end
function a! end
dimension() = nothing
=#
include("utility_functions.jl")
include("containers.jl")
include("guided_proposals.jl")
include("ode_solver_general.jl")
include("ode_solver_hfc.jl")
include("ode_solver_mlmu.jl")
include("ode_solver_pnu.jl")


export GuidProp, H, F, c

end # module
