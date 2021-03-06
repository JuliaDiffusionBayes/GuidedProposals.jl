module GuidedProposals

    using DiffEqCallbacks, OrdinaryDiffEq, LinearAlgebra
    using StaticArrays, Random, RecursiveArrayTools, Trajectories
    using DiffusionDefinition
    using ObservationSchemes
    const DD = DiffusionDefinition
    const OBS = ObservationSchemes
    import DiffusionDefinition: set_parameters!, same_entries

    include("utility_functions.jl")
    include("containers.jl")
    include("best_containers.jl")
    include("ode_solver_general.jl")
    include("guided_proposals.jl")
    include("log_likelihood.jl")
    include("sampling.jl")
    include("reparameterizations.jl")
    include("ode_solver_hfc.jl")
    include("ode_solver_mlmu.jl")
    include("ode_solver_pnu.jl")

    export GuidProp, H, F, c, recompute_guiding_term!, build_guid_prop
    export loglikhd, loglikhd_obs
    export forward_guide!, backward_filter!
    export guid_prop_for_blocking, set_obs!
    export standard_guid_prop_time_transf
end
