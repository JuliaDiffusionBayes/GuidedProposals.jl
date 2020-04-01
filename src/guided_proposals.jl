#==============================================================================#
#                                                                              #
#              Contains the main struct of the package `GuidProp`              #
#              that defines a sampler for conditioned diffusions               #
#                                                                              #
#==============================================================================#
"""
    struct GuidProp{
            K,DP,DW,SS,R,R2,O,S,T
            } <: DiffusionDefinition.DiffusionProcess{K,DP,DW,SS}
        P_target::R
        P_aux::R2
        obs::O
        guiding_term_solver::S
    end

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
                    solver=Tsit5(),
                    ode_type=:HFc,
                    convert_to_HFc=false,
                    mode=:inplace,
                    gradients=false,
                    eltype=Float64,
                ),
                next_guiding_term=nothing
            ) where {R,R2,O}

    Default constructor. `P_target` and `P_aux` are the target and auxiliary
    diffusion laws respectively. `tt` is the time-grid on which `∇logρ` needs
    to be computed. `obs` is the terminal observation (and the only one on the
    interval (`tt[1]`, `tt[end]`]). `solver_choice` specifies the type of ODE
    solver that is to be used for computations of `∇logρ`
        ( it is a `NamedTuple`, where `solver` specifies the algorithm for
        solving ODEs (see the documentation of DifferentialEquations.jl for
        possible choices), `ode_type` picks the ODE system (between :HFc, :MLμ
        and :Pν), `convert_to_HFc` indicates whether to translate the results of
        M,L,μ solver to H,F,c objects, `mode` is a flag indicating the way in
        which data is being handled:
        - `:inplace`: uses regular arrays to store the data (requires functions
                      B!, β!, σ! and a! to be defined)
        - `:outofplace`: operates on static arrays
        - `:gpu`: operates on GPU arrays [TODO not implemented yet]
        `gradients` is a flag indicating whether automatic differentiation is to
        be employed and `eltype` indicates the data-type of each container's
        member. )
    Finally, `next_guiding_term` is the guided proposal for the subsequent
    inter-observation interval.
"""
struct GuidProp{
        K,DP,DW,SS,R,R2,O,S,T
        } <: DiffusionDefinition.DiffusionProcess{K,DP,DW,SS}
    P_target::R
    P_aux::R2
    obs::O
    guiding_term_solver::S

    function GuidProp(
            tt,
            P_target::R,
            P_aux::R2,
            obs::O,
            solver_choice=(
                solver=Tsit5(),
                ode_type=:HFc,
                convert_to_HFc=false,
                mode=:inplace,
                gradients=false,
                eltype=Float64,
            ),
            next_guiding_term=nothing
        ) where {R,R2,O}

        choices_now, choices_to_pass_on = reformat(
            solver_choice,
            next_guiding_term===nothing,
            P_aux,
        )

        params = (
            collect(tt),
            P_aux,
            obs,
            Val{choices_now.mode}(),
            choices_to_pass_on,
            next_guiding_term,
        )
        guiding_term_solver = init_solver(
            Val{choices_now.ode_type}(),
            Val{choices_now.convert_to_HFc}(),
            params...
        )
        S = typeof(guiding_term_solver)
        T = choices_now.ode_type

        DP, DW = DD.dimension(P_target)
        K, SS = eltype(P_target), DD.state_space(P_target)

        new{K,DP,DW,SS,R,R2,O,S,T}(P_target, P_aux, obs, guiding_term_solver)
    end
end

"""
    reformat(solver_choice::NamedTuple, last_interval::Bool, P_aux)

Re-format the `solver_choice` by splitting it into two NamedTuples and
populating any missing entries with defaults. `P_aux` is the law of the
auxiliary diffusion that is needed in case a flag for computing gradients is
turned on.
"""
function reformat(solver_choice::NamedTuple, last_interval::Bool, P_aux)
    # `get_or_default` is defined in `utility_functions.jl`
    choices_needed_later = (
        solver = get_or_default(solver_choice, :solver, Tsit5()),
        eltype = fetch_eltype(solver_choice, P_aux),
    )
    choices_needed_now = (
        ode_type = lowercase(get_or_default(solver_choice, :ode_type, :HFc)),
        mode = lowercase(get_or_default(solver_choice, :mode, :inplace)),
        gradients = get_or_default(solver_choice, :gradients, false),
        convert_to_HFc = get_or_default(solver_choice, :convert_to_HFc, false),
    )
    # Using MLμ solver as a helper for deriving H,F,c can be done only on the
    # terminal interval.
    @assert ( last_interval || choices_needed_now.convert_to_HFc==false)
    @assert choices_needed_now.ode_type in [:hfc, :mlμ, :pν]
    @assert choices_needed_now.mode in [:inplace,:outofplace,:gpu]
    @assert typeof(choices_needed_later.eltype) <: DataType

    choices_needed_now, choices_needed_later
end

"""
    fetch_eltype(choices, P)

Determine the type of the elements that is supposed to be used by the internal
containers of this package. If `choice.gradients` flag is turned on, then
use the same type as the eltypes of the parameters in the auxiliary law.
Otherwise, use the type specified in `choice`. If neither the `choice.gradients`
flag is on, nor a default is provided, use Float64.
"""
function fetch_eltype(choices, P)
    (
        (haskey(choices, :gradients) && choices.gradients) ?
        eltype(P) :
        get_or_default(choices, :eltype, Float64)
    )
end


"""
    init_solver(
        ::Val{:hfc},
        ::Any,
        tt,
        P_aux,
        obs,
        mode::Val,
        choices,
        next_guiding_term
    )

Initialise ODE solver for H,F,c, preallocate space and solve it once. `tt` is
the time-grid on which `∇logρ` is to be saved. `P_aux` is the auxiliary law,
`obs` is the terminal observation, `mode` is for differentiating between
in-place, out-of-place and gpu constructors for the guiding term solver,
`choices` contains additional information that is passed on (and which is about
eltype and a chosen algorithm for the ODE solvers) and finally,
`next_guiding_term` is the guided proposal used on the subsequent
inter-observation interval.
"""
function init_solver(
        ::Val{:hfc},
        ::Any,
        tt,
        P_aux,
        obs,
        mode::Val,
        choices,
        next_guiding_term
    )

    xT_plus = fetch_xT_plus(
        mode,
        next_guiding_term,
        choices.eltype,
        DD.dimension(P_aux).process
    )

    HFcSolver(
        mode,
        tt,
        xT_plus,
        P_aux,
        obs,
        choices,
    )
end

"""
    fetch_xT_plus(::Val{:inplace}, next_guiding_term, el, dim_of_proc)

If this is not the last inter-observation interval, fetch the data containing
H,F,c computed for the left time-limit of the subsequent interval. Otherwise,
instantiate a zero-term.
"""
function fetch_xT_plus end

#TODO switch to using buffers for this step
function fetch_xT_plus(::Val{:inplace}, next_guiding_term, el, dim_of_proc)
    (
        next_guiding_term===nothing ?
        HFcContainer{el}(dim_of_proc) :
        HFc0(next_guiding_term)
    )
end

function fetch_xT_plus(::Val{:outofplace}, next_guiding_term, el, dim_of_proc)
    (
        next_guiding_term===nothing ?
        zero(SVector{size_of_HFc_solution(dim_of_proc),el}) :
        HFc0(next_guiding_term)
    )
end

"""
    size_of_HFc_solution(d)

Compute the size of a vector containing H,F,c elements
"""
size_of_HFc_solution(d) = d^2+d+1

"""
    size_of_HFc_solution(d)

Length of a vector containing temporary data needed for in-place solver of
H,F,c, when the underlying process has dimension `d`.
"""
size_of_HFc_buffer(d) = 4*d^2+d

#TODO
init_solver(::Val{:mlμ}, args...) = nothing

#TODO
init_solver(::Val{:Pν}, args...) = nothing

"""
    HFc0(P::GuidProp)

Return the container with data that can be used to reconstruct H,F,c evaluated
at time 0+ for the guided proposal `P`.
"""
HFc0(P::GuidProp) = HFc0(P.guiding_term_solver)

"""
    H(P::GuidProp, i)

Return saved matrix H[i] (with H[1] indicating H at time 0+ and H[end]
indicating H at time T).
"""
H(P::GuidProp, i) = H(P.guiding_term_solver, i)

"""
    F(P::GuidProp, i)

Return saved vector F[i] (with F[1] indicating F at time 0+ and F[end]
indicating F at time T).
"""
F(P::GuidProp, i) = F(P.guiding_term_solver, i)

"""
    c(P::GuidProp, i)

Return saved scalar c[i] (with c[1] indicating c at time 0+ and c[end]
indicating c at time T).
"""
c(P::GuidProp, i) = c(P.guiding_term_solver, i)

"""
    Base.time(P::GuidProp, i)

Return time-point tt[i] corresponding to a saved state of ODEs (with tt[1]
indicating time 0+ and tt[end] indicating time T).
"""
Base.time(P::GuidProp, i) = P.guiding_term_solver.saved_values.t[end-i+1]

DD.dimension(P::GuidProp) = DD.dimension(P.P_target)

abstract type IntegrationRule end

struct LeftRule <: IntegrationRule end

mode(P::GuidProp) =  mode(P.guiding_term_solver)

function recompute_guiding_term(P::GuidProp, next_guiding_term=nothing)
    xT_plus = fetch_xT_plus(
        Val{mode(P)}(),
        next_guiding_term,
        eltype(HFc0(P)),
        DD.dimension(P).process
    )
    recompute_guiding_term(P.guiding_term_solver, P.P_aux, P.obs, xT_plus)
end

DD.constdiff(P::GuidProp) = DD.constdiff(P.P_target) && DD.constdiff(P.P_aux)


DD._σ((t,i)::DD.IndexedTime, x, P::GuidProp) = DD.σ((t,i), x, P.P_target)

DD._b((t, i)::DD.IndexedTime, x, P::GuidProp) = (
    DD.b(t, x, P.P_target)
    + DD.a(t, x, P.P_target) * ∇logρ(i, x, P)
)

function DD._b!(buffer, (t, i)::DD.IndexedTime, x, P::GuidProp)
    DD.b!(buffer.b, t, x, P.P_target)
    DD.a!(buffer.a, t, x, P.P_target)
    ∇logρ!(buffer, i, x, P)
    mul!(buffer.b, buffer.a, buffer.∇logρ, true, true)
end

∇logρ(i::Integer, x, P::GuidProp) = ∇logρ(i, x, P.guiding_term_solver)


function loglikelihood(X::Trajectory, P::GuidProp; skip=0)
    loglikelihood(LeftRule(), X, P, P.guiding_term_solver; skip=skip)
end

function loglikelihood(ir::IntegrationRule, X::Trajectory, P::GuidProp; skip=0)
    loglikelihood(ir, X, P, P.guiding_term_solver; skip=skip)
end

function loglikelihood(
        ::LeftRule,
        X::Trajectory,
        P::GuidProp,
        ::AbstractGuidingTermSolver{:outofplace};
        skip=0
    )
    tt, xx = X.t, X.x
    ll = 0.0
    N = length(tt)-1-skip

    for i in 1:N
        x, s, dt = xx[i], tt[i], tt[i+1]-tt[i]
        r_i = ∇logρ(i, x, P)
        b_i = DD.b(s, x, P.P_target)
        btil_i = DD.b(s, x, P.P_aux)

        ll += dot(b_i-btil_i, r_i) * dt

        if !DD.constdiff(P)
            H_i = H(i, x, P)
            a_i = DD.a(s, x, P.P_target)
            atil_i = DD.a(s, x, P.P_aux)
            ll -= 0.5*sum( (a_i - atil_i).*H_i ) * dt
            ll += 0.5*( r_i'*(a_i - atil_i)*r_i ) * dt
        end
    end
    ll
end

# NOTE worry about this later...
function loglikelihood(
        ::LeftRule,
        X::Trajectory,
        P::GuidProp,
        ::AbstractGuidingTermSolver{:inplace};
        skip=0
    )
    tt, xx = X.t, x.x
    ll = 0.0
    N = length(tt)-1-skip

    for i in 1:N
        x, s, dt = xx[i], tt[i], tt[i+1]-tt[i]
        ∇logρ!(P.buffer.r_i, i, x, P)
        DD.b(P.buffer.b_i, s, x, P.P_target)
        DD.b(s.buffer.btil_i, x, P.P_aux)
        for j in 1:length(P.buffer.b_i)
            P.buffer.b_i[j] -= P.buffer.btil_i[j]
        end

        ll += dot(P.buffer.b_i, P.buffer.r_i) * dt

        if !DD.constdiff(P)
            H_i = H(i, x, P)
            DD.a(P.buffer.a_i, s, x, P.P_target)
            DD.a(P.buffer.atil_i, s, x, P.P_aux)
            ll -= 0.5*sum( (a_i - atil_i).*H_i ) * dt
            ll += 0.5*( r_i'*(a_i - atil_i)*r_i ) * dt
        end
    end
    ll
end

loglikelihood_obs(P::GuidProp, x0) = loglikelihood_obs(P, x0, P.guiding_term_solver)

function loglikelihood_obs(P::GuidProp, x0, ::AbstractGuidingTermSolver{:outofplace})
    - 0.5 * ( x0'*H(P, 1)*x0 - 2.0*dot(F(P, 1), x0) ) - c(P, 1)
end

# worry about it later
function loglikelihood_obs(P::GuidProp, x0, ::AbstractGuidingTermSolver{:inplace})
    - 0.5 * ( x0'*H(P, 1)*x0 - 2.0*dot(F(P, 1), x₀) ) - c(P, 1)
end
