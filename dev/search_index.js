var documenterSearchIndex = {"docs":
[{"location":"#GuidedProposals.jl-1","page":"Home","title":"GuidedProposals.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"test documentation","category":"page"},{"location":"#","page":"Home","title":"Home","text":"","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Modules = [GuidedProposals]","category":"page"},{"location":"#GuidedProposals.GuidProp","page":"Home","title":"GuidedProposals.GuidProp","text":"struct GuidProp{\n        K,DP,DW,SS,R,R2,O,S,T\n        } <: DiffusionDefinition.DiffusionProcess{K,DP,DW,SS}\n    P_target::R\n    P_aux::R2\n    obs::O\n    guiding_term_solver::S\nend\n\nStruct defining guided proposals of M Schauer, F van der Meulen and H van Zanten. See Mider M, Schauer M and van der Meulen F Continuous-discrete smoothing of diffusions (2020) for a comprehensive overview of the mathematics behind this object. It computes and stores the guiding term ∇logρ and allows for simulation of guided proposals and computation of their likelihood.\n\n    GuidProp(\n            tt,\n            P_target::R,\n            P_aux::R2,\n            obs::O,\n            solver_choice=(\n                solver=Tsit5(),\n                ode_type=:HFc,\n                convert_to_HFc=false,\n                mode=:inplace,\n                gradients=false,\n                eltype=Float64,\n            ),\n            next_guided_prop=nothing\n        ) where {R2,O}\n\nDefault constructor. `P_target` and `P_aux` are the target and auxiliary\ndiffusion laws respectively, `tt` is the time-grid on which `∇logρ` needs to\nbe computed. `obs` is the terminal observation (and the only one on the\ninterval (`tt[1]`, `tt[end]`]). `solver_choice` specifies the type of ODE\nsolver that is to be used for computations of `∇logρ`\n    ( it is a `NamedTuple`, where `solver` specifies the algorithm for\n    solving ODEs (see the documentation of DifferentialEquations.jl for\n    possible choices), `ode_type` picks the ODE system (between :HFc, :MLμ\n    and :Pν), `convert_to_HFc` indicates whether to translate the results of\n    M,L,μ solver to H,F,c objects, `mode` is a flag indicating the way in\n    which data is being handled:\n    - `:inplace`: uses regular arrays to store the data (requires functions\n                  B!, β!, σ! and a! to be defined)\n    - `:outofplace`: operates on static arrays\n    - `:gpu`: operates on GPU arrays [TODO not implemented yet]\n    `gradients` is a flag indicating whether automatic differentiation is to\n    be employed and `eltype` indicates the data-type of each container's\n    member. )\nFinally, `next_guided_prop` is the guided proposal for the subsequent\ninter-observation interval.\n\n\n\n\n\n","category":"type"},{"location":"#GuidedProposals.GuidProp-Tuple{GuidProp,Any,Any}","page":"Home","title":"GuidedProposals.GuidProp","text":"GuidProp(P::GuidProp, θ°, η°)\n\nCreate new guided proposals object with new parameters θ° parametrizing the diffusion laws and η° parametrizing the observations. Keep the pre-allocated spaces for solvers unchanged.\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.F-Tuple{GuidProp,Any}","page":"Home","title":"GuidedProposals.F","text":"F(P::GuidProp, i)\n\nReturn saved vector Fi.\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.F-Tuple{GuidedProposals.HFcSolver,Integer}","page":"Home","title":"GuidedProposals.F","text":"F(s::HFcSolver, i::Integer)\n\nReturn saved vector Fi.\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.H-Tuple{GuidProp,Any}","page":"Home","title":"GuidedProposals.H","text":"H(P::GuidProp, i)\n\nReturn saved matrix Hi.\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.H-Tuple{GuidedProposals.HFcSolver,Integer}","page":"Home","title":"GuidedProposals.H","text":"H(s::HFcSolver, i::Integer)\n\nReturn saved matrix Hi.\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.c-Tuple{GuidProp,Any}","page":"Home","title":"GuidedProposals.c","text":"c(P::GuidProp, i)\n\nReturn saved scalar ci.\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.c-Tuple{GuidedProposals.HFcSolver,Integer}","page":"Home","title":"GuidedProposals.c","text":"c(s::HFcSolver, i::Integer)\n\nReturn saved scalar ci.\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.recompute_guiding_term!","page":"Home","title":"GuidedProposals.recompute_guiding_term!","text":"recompute_guiding_term!(P::GuidProp, next_guided_prop=nothing)\n\nRecompute the guiding term (most often used after update of parameters or change of an observation). next_guided_prop is the guided proposal law from the subsequent interval\n\n\n\n\n\n","category":"function"},{"location":"#GuidedProposals.AbstractGuidingTermSolver","page":"Home","title":"GuidedProposals.AbstractGuidingTermSolver","text":"AbstractGuidingTermSolver{Tmode}\n\nSupertype for ODE solvers (solving H,F,c system or M,L,μ system or P,ν system). Tmode is a flag for whether computations are done in-place (with states represented by vectors), out-of-place (with state represented by StaticArrays), or on GPUs (with states represented by cuArrays).\n\n\n\n\n\n","category":"type"},{"location":"#GuidedProposals.HFcBuffer","page":"Home","title":"GuidedProposals.HFcBuffer","text":"struct HFcBuffer{\n        T,D,TB,Tβ,Tσ,Ta,Tmat,Tvec\n        } <: DiffusionDefinition.AbstractBuffer{T}\n    data::Vector{T}\n    B::TB\n    β::Tβ\n    σ::Tσ\n    a::Ta\n    mat::Tmat\n    vec::Tvec\nend\n\nA buffer for temporary computations of in-place ODE solvers solving for H,F,c system.\n\n\n\n\n\n","category":"type"},{"location":"#GuidedProposals.HFcContainer","page":"Home","title":"GuidedProposals.HFcContainer","text":"struct HFcContainer{T,D,TH,TF,Tc} <: AbstractBuffer{T}\n    data::Vector{T}\n    H::TH\n    F::TF\n    c::Tc\nend\n\nA buffer containing data for in-place computations of H,F,c terms.\n\n\n\n\n\n","category":"type"},{"location":"#GuidedProposals.HFcSolver","page":"Home","title":"GuidedProposals.HFcSolver","text":"struct HFcSolver{Tmode,Tsv,Tps,Tcb,Ts,T,Ta} <: AbstractGuidingTermSolver{Tmode}\n    saved_values::Tsv\n    problem_setup::Tps\n    callback::Tcb\n    solver::Ts\n    HFc0::T\n    access::Ta\nend\n\nStruct for solving a (H,F,c)-system of ODEs.\n\n    HFcSolver(\n        ::Val{:inplace},\n        tt,\n        xT_plus,\n        P,\n        obs,\n        choices,\n    )\nConstructor for an ODE solver with in-place operations. Pre-allocates space\nand solves a (H,F,c)-system of ODEs once, backward in time on the interval\n`(tt[1], tt[end])`, with a terminal condition computed from `xT_plus`. `P`\nis the auxiliary diffusion law, `obs` is the observation made at time\n`tt[end]` and `choices` contains information about eltype and the algorithm\nfor solving ODEs. H,F,c are saved on a grid of time-points `tt`.\n\n    HFcSolver(\n        ::Val{:outofplace},\n        tt,\n        xT_plus,\n        P,\n        obs,\n        choices\n    )\nConstructor for an ODE solver with out-of-place operations using\nStaticArrays. Initialises the object and solves a (H,F,c)-system of ODEs\nonce, backward in time on the interval `(tt[1], tt[end])`, with a terminal\ncondition computed from `xT_plus`. `P` is the auxiliary diffusion law, `obs`\nis the observation made at time `tt[end]` and `choices` contains information\nabout eltype and the algorithm for solving ODEs. H,F,c are saved on a grid\nof time-points `tt`.\n\n\n\n\n\n","category":"type"},{"location":"#GuidedProposals.IntegrationRule","page":"Home","title":"GuidedProposals.IntegrationRule","text":"IntegrationRule\n\nSupertype of all integration rules. In this package we currently approximate the integrals by step functions with evaluations at the left side of the intervals.\n\n\n\n\n\n","category":"type"},{"location":"#GuidedProposals.LeftRule","page":"Home","title":"GuidedProposals.LeftRule","text":"LeftRule <: IntegrationRule\n\nIntegration rule flag, indicating to approximate functions with step functions with height equal to function evaluation at the left side of the intevals.\n\n\n\n\n\n","category":"type"},{"location":"#Base.Libc.time-Tuple{GuidProp,Any}","page":"Home","title":"Base.Libc.time","text":"Base.time(P::GuidProp, i)\n\nReturn time-point tt[i] corresponding to a saved state of ODEs (with tt[1] indicating time 0+ and tt[end] indicating time T).\n\n\n\n\n\n","category":"method"},{"location":"#DiffusionDefinition.constdiff-Tuple{GuidProp}","page":"Home","title":"DiffusionDefinition.constdiff","text":"\n\n\n\n","category":"method"},{"location":"#DiffusionDefinition.dimension-Tuple{GuidProp}","page":"Home","title":"DiffusionDefinition.dimension","text":"DD.dimension(P::GuidProp)\n\nDimension of the stochastic process and the driving Brownian motion (by default the same as that of the target process)\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.HFc0-Tuple{GuidProp}","page":"Home","title":"GuidedProposals.HFc0","text":"HFc0(P::GuidProp)\n\nReturn the container with data that can be used to reconstruct H,F,c evaluated at time 0+ for the guided proposal P.\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.HFc0-Tuple{GuidedProposals.HFcSolver}","page":"Home","title":"GuidedProposals.HFc0","text":"HFc0(s::HFcSolver)\n\nReturn the data containing H,F,c terms for the time 0+\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.fetch_eltype-Tuple{Any,Any}","page":"Home","title":"GuidedProposals.fetch_eltype","text":"fetch_eltype(choices, P)\n\nDetermine the type of the elements that is supposed to be used by the internal containers of this package. If choice.gradients flag is turned on, then use the same type as the eltypes of the parameters in the auxiliary law. Otherwise, use the type specified in choice. If neither the choice.gradients flag is on, nor a default is provided, use Float64.\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.fetch_xT_plus","page":"Home","title":"GuidedProposals.fetch_xT_plus","text":"fetch_xT_plus(::Val{:inplace}, next_guided_prop, el, dim_of_proc)\n\nIf this is not the last inter-observation interval, fetch the data containing H,F,c computed for the left time-limit of the subsequent interval. Otherwise, instantiate a zero-term.\n\n\n\n\n\n","category":"function"},{"location":"#GuidedProposals.get_or_default-Tuple{Any,Symbol,Any}","page":"Home","title":"GuidedProposals.get_or_default","text":"get_or_default(container, elem::Symbol, default)\n\nReturn container.elem if it exists, otherwise return default\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.init_solver-Tuple{Val{:hfc},Any,Any,Any,Any,Val,Any,Any}","page":"Home","title":"GuidedProposals.init_solver","text":"init_solver(\n    ::Val{:hfc},\n    ::Any,\n    tt,\n    P_aux,\n    obs,\n    mode::Val,\n    choices,\n    next_guided_prop\n)\n\nInitialise ODE solver for H,F,c, preallocate space and solve it once. tt is the time-grid on which ∇logρ is to be saved. P_aux is the auxiliary law, obs is the terminal observation, mode is for differentiating between in-place, out-of-place and gpu constructors for the guiding term solver, choices contains additional information that is passed on (and which is about eltype and a chosen algorithm for the ODE solvers) and finally, next_guided_prop is the guided proposal used on the subsequent inter-observation interval.\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.mode-Tuple{GuidProp}","page":"Home","title":"GuidedProposals.mode","text":"mode(P::GuidProp)\n\nReturn the mode of solving ODE systems (:inplace, :outofplace or :gpu) [TODO not used much, for multiple dispatch needs to return Val{mode}() instead, change  or remove].\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.outer-Tuple{Any}","page":"Home","title":"GuidedProposals.outer","text":"outer(x)\n\nCompute an outer product\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.prepare_static_saving_types-Union{Tuple{D}, Tuple{Val{:hfc},Val{D},Any}} where D","page":"Home","title":"GuidedProposals.prepare_static_saving_types","text":"prepare_static_saving_types(::Val{:hfc}, ::Val{D}, el) where D\n\nDefine data-types for H,F,c computed by out-of-place solver that are to be saved internally.\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.reformat-Tuple{NamedTuple,Bool,Any}","page":"Home","title":"GuidedProposals.reformat","text":"reformat(solver_choice::NamedTuple, last_interval::Bool, P_aux)\n\nRe-format the solver_choice by splitting it into two NamedTuples and populating any missing entries with defaults. P_aux is the law of the auxiliary diffusion that is needed in case a flag for computing gradients is turned on.\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.size_of_HFc_buffer-Tuple{Any}","page":"Home","title":"GuidedProposals.size_of_HFc_buffer","text":"size_of_HFc_solution(d)\n\nLength of a vector containing temporary data needed for in-place solver of H,F,c, when the underlying process has dimension d.\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.size_of_HFc_solution-Tuple{Any}","page":"Home","title":"GuidedProposals.size_of_HFc_solution","text":"size_of_HFc_solution(d)\n\nCompute the size of a vector containing H,F,c elements\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.static_accessor_HFc-Union{Tuple{T}, Tuple{K}, Tuple{K,Val{T}}} where T where K<:Union{StaticArrays.MArray{Tuple{S},T,1,S} where T where S, StaticArrays.SArray{Tuple{S},T,1,S} where T where S}","page":"Home","title":"GuidedProposals.static_accessor_HFc","text":"static_accessor_HFc(u::SVector, ::Val{T}) where T\n\nAccess data stored in the container u so that it matches the shapes of H,F,c and points to the correct points in u. T is the dimension of the stochastic process.\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.update_HFc!-Tuple{Any,Any,Any}","page":"Home","title":"GuidedProposals.update_HFc!","text":"update_HFc!(u_T, u_Tplus, obs, access)\n\nUpdate equations for H,F,c at the times of observations. Save the data into u_T.\n\n\n\n\n\n","category":"method"},{"location":"#GuidedProposals.update_HFc-Tuple{Any,Any,Any}","page":"Home","title":"GuidedProposals.update_HFc","text":"update_HFc(u_Tplus, obs, access)\n\nUpdate equations for H,F,c at the times of observations.\n\n\n\n\n\n","category":"method"}]
}
