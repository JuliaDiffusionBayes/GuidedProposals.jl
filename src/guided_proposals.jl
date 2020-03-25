#==============================================================================#
#                                                                              #
#              Contains the main struct of the package `GuidProp`              #
#              that defines a sampler for conditioned diffusions               #
#                                                                              #
#==============================================================================#

# allow for a switch between MLμ and other solvers only at the terminal
# observation
"""
    GuidProp{R,R2,O,C,SM,SH,SP}

Struct defining `guided proposals` of M Schauer, F van der Meulen and H van
Zanten. See Mider M, Schauer M and van der Meulen F `Continuous-discrete
smoothing of diffusions` (2020) for a comprehensive overview of the mathematics
behind this object. It computes and stores the guiding term `∇logρ` and allows
for simulation of guided proposals and computation of their likelihood.

        GuidProp(
            tt,
            P_target::R,
            P_aux::R2,
            obs::O,
            solver_choice=(
                solver=:Tsit5,
                ode_type=:HFc,
                convert_to_HFc=false,
                inplace=false,
                save_as_type=nothing,
                ode_data_type=nothing,
            ),
            next_guiding_term=nothing
        ) where {R,R2,O}

    Default constructor. `P_target` and `P_aux` are the target and auxiliary
    diffusion laws respectively. `tt` is the time-grid on which `∇logρ` needs
    to be computed. `obs` is the terminal observation (and the only one on the
    interval (`tt[1]`, `tt[end]`]). `solver_choice` specifies the type of ODE
    solver that is to be used for computations of `∇logρ` (it is a `NamedTuple`,
    with `solver` specifying the ODE solver, `ode_type` picking the ODE system
    (between :HFc, :MLμ and :Pν), `convert_to_HFc` indicating whether to
    translate the results of M,L,μ solver to H,F,c objects, `inplace` is a flag
    for whether to use in-place ODE solvers (for which B!, β!, σ! and a!
    functions need to be defined for the auxiliary diffusion), `save_as_type`
    specifies the datatypes in which the terms computed for `∇logρ` are to be
    saved and `ode_data_type` specifies the data type used for internal
    computations of ODE solvers). Finally, `next_guiding_term` is the guided
    proposal for the subsequent inter-observation interval.
"""
struct GuidProp{R,R2,O,T,SM,SH,SP}
    P_target::R
    P_aux::R2
    obs::O
    MLμ::SM
    HFc::SH
    Pν::SP

    function GuidProp(
            tt,
            P_target::R,
            P_aux::R2,
            obs::O,
            solver_choice=(
                solver=Tsit5(),
                ode_type=:HFc,
                convert_to_HFc=false,
                inplace=false,
                save_as_type=nothing,
                ode_data_type=nothing,
            ),
            next_guiding_term=nothing
        ) where {R,R2,O}
        params = (
            collect(tt),
            P_aux,
            obs,
            solver_choice,
            next_guiding_term,
        )
        SM, SH, SP, MLμ, HFc, Pν = init_solvers(params)
        # define a helper flag for accessors to H,F,c
        HFc_access_path = (
            (solver_choice.convert_to_HFc && (lowercase(ode_type)==:mlμ) ) ?
            :via_mlμ :
            :via_hfc
        )
        new{R,R2,O,HFc_access_path,SM,SH,SP}(P_target, P_aux, obs, MLμ, HFc, Pν)
    end
end

"""
    init_solvers(params)

Initialise appropriate ODE solvers, according to how they are specified in
`params`. Validity of the format of `params` is not checked, because it is only
ever called by hard-coded constructor of `GuidProp`.
"""
function init_solvers(params)
    sol, next_gp = params[4], params[5]
    # Using MLμ solver as a helper to a primary solver (of other type) used
    # on any other interval can be done only on the terminal interval.
    @assert ( next_gp===nothing || sol.convert_to_HFc==false )

    ode_choice = lowercase(sol.ode_type)
    @assert ode_choice in [:hfc, :mlμ, :pν]

    MLμ = init_mlμ(
        Val{ode_choice == :mlμ}(),
        Val{sol.convert_to_HFc}(),
        params...
    )
    HFc = init_hfc(
        Val{ode_choice == :hfc}(),
        params...
    )
    Pν = init_pν(
        Val{ode_choice == :pν}(),
        params...
    )
    typeof(MLμ), typeof(HFc), typeof(Pν), MLμ, HFc, Pν
end

"""
    init_hfc(::Val{false}, ...)

Nothing to initialise if a flag for HFc solver is turned off
"""
init_hfc(::Val{false}, args...) = nothing

"""
    init_hfc(::Val{true}, tt, P_aux, obs, choice, next_guiding_term)

Initialise ODE solver for H,F,c, preallocate space and solve it once. `tt` is
the time-grid on which `∇logρ` is to be saved. `P_aux` is the auxiliary law,
`obs` is the terminal observation, `choice` is the `NamedTuple` with various
choices for the the ODE solver (here, the ones used are: `solver`---the type of
ODE solver, `inplace`---whether to perform computations in-place,
`save_as_type`---the datatypes in which H,F,c are to be solved (if set to
`nothing`, then Matrix{Float64}, Vector{Float64} and Float64 are used by default
for storing H, F and c respectively), `ode_data_type`---the datatypes in which
the ODE solvers should store all the data (if set to `nothing`, then
Vector{Float64} is used by default)). Finally, `next_guiding_term` is the guided
proposal used on the subsequent inter-observation interval.
"""
function init_hfc(::Val{true}, tt, P_aux, obs, choice, next_guiding_term)
    d = dimension(P_aux).process
    inplace = Val{choice.inplace ? :inplace : :outofplace}()
    xT_plus = (
        next_guiding_term===nothing ?
        init_xT_plus(Val{:hfc}(), inplace, d, choice.ode_data_type) :
        HFc0(next_guiding_term)
    )

    HFcSolver(
        inplace,
        tt,
        xT_plus,
        P_aux,
        obs,
        choice.solver,
        choice.save_as_type
    )
end

HFc0(P::GuidProp{R,R2,O,C,T}) where {R,R2,O,C,T<:Val{:via_hfc}} = HFc0(P.HFc)
HFc0(P::GuidProp{R,R2,O,C,T}) where {R,R2,O,C,T<:Val{:via_mlμ}} = HFc0(P.MLμ)

function init_xT_plus(
        ::Val{:hfc},
        ::Val{:inplace},
        dim_of_process,
        ode_data_type
    )
    ode_data_type = ( ode_data_type===nothing ? Float64 : ode_data_type )
    zeros(ode_data_type, size_of_HFc_solution(dim_of_process))
end

function init_xT_plus(
        ::Val{:hfc},
        ::Val{:outofplace},
        dim_of_process,
        ode_data_type
    )
    ode_data_type = ( ode_data_type===nothing ? Float64 : ode_data_type )
    zeros(ode_data_type, dim_of_process)
end

size_of_HFc_solution(d) = d^2+d+1
size_of_HFc_buffer(d) = 4*d^2+d

init_mlμ(::Any, args...) = nothing

init_pν(::Any, args...) = nothing

H(P::GuidProp{R,R2,O,:via_mlμ}, i) where {R,R2,O} = H(P.MLμ, i)
F(P::GuidProp{R,R2,O,:via_mlμ}, i) where {R,R2,O} = F(P.MLμ, i)
c(P::GuidProp{R,R2,O,:via_mlμ}, i) where {R,R2,O} = c(P.MLμ, i)
HFc0(P::GuidProp{R,R2,O,:via_mlμ}) where {R,R2,O} = error("not implemented")

H(P::GuidProp{R,R2,O,:via_hfc}, i) where {R,R2,O} = H(P.HFc, i)
F(P::GuidProp{R,R2,O,:via_hfc}, i) where {R,R2,O} = F(P.HFc, i)
c(P::GuidProp{R,R2,O,:via_hfc}, i) where {R,R2,O} = c(P.HFc, i)
HFc0(P::GuidProp{R,R2,O,:via_hfc}) where {R,R2,O} = P.HFc.HFc0
