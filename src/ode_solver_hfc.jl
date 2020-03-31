#==============================================================================#
#
#       ODE solver for H,F,c terms; relies on DifferentialEquations.jl
#
#==============================================================================#
"""
    struct HFcSolver{Tmode,Tsv,Tps,Tcb,Ts,T,Ta} <: AbstractGuidingTermSolver{Tmode}
        saved_values::Tsv
        problem_setup::Tps
        callback::Tcb
        solver::Ts
        HFc0::T
        access::Ta
    end

Struct for solving a (H,F,c)-system of ODEs.

        HFcSolver(
            ::Val{:inplace},
            tt,
            xT_plus,
            P,
            obs,
            choices,
        )
    Constructor for an ODE solver with in-place operations. Pre-allocates space
    and solves a (H,F,c)-system of ODEs once, backward in time on the interval
    `(tt[1], tt[end])`, with a terminal condition computed from `xT_plus`. `P`
    is the auxiliary diffusion law, `obs` is the observation made at time
    `tt[end]` and `choices` contains information about eltype and the algorithm
    for solving ODEs. H,F,c are saved on a grid of time-points `tt`.

        HFcSolver(
            ::Val{:outofplace},
            tt,
            xT_plus,
            P,
            obs,
            choices
        )
    Constructor for an ODE solver with out-of-place operations using
    StaticArrays. Initialises the object and solves a (H,F,c)-system of ODEs
    once, backward in time on the interval `(tt[1], tt[end])`, with a terminal
    condition computed from `xT_plus`. `P` is the auxiliary diffusion law, `obs`
    is the observation made at time `tt[end]` and `choices` contains information
    about eltype and the algorithm for solving ODEs. H,F,c are saved on a grid
    of time-points `tt`.
"""
struct HFcSolver{Tmode,Tsv,Tps,Tcb,Ts,T,Ta} <: AbstractGuidingTermSolver{Tmode}
    saved_values::Tsv
    problem_setup::Tps
    callback::Tcb
    solver::Ts
    HFc0::T
    access::Ta

    function HFcSolver(
            ::Val{:inplace},
            tt,
            xT_plus,
            P,
            obs,
            choices,
        )
        function HFc_update!(du, u, p, t)
            # shorthand names for views, hopefully optimised
            # ---
            # current state
            H, F, c = u.H, u.F, u.c
            B, β, σ, a, tmat, tvec = p.B, p.β, p.σ, p.a, p.mat, p.vec
            # increments (to-be-computed by this function)
            dH, dF, dc = du.H, du.F, du.c
            # ---

            # in-place computation of auxiliary process; stored in du
            DD.B!(B, t, P), DD.β!(β, t, P), DD.σ!(σ, t, P), DD.a!(a, t, P)

            # ODEs
            # ---
            # dH = - (_B' * H) - (H * _B) + outer(H * σ)
            mul!(tmat, H, a)
            mul!(dH, tmat, tmat')
            mul!(dH, H, B, -true, true)
            mul!(dH, B', H, -true, true)
            # dF = - (_B' * F) + (H * a * F) + (H * _β)
            mul!(dF, tmat, F)
            mul!(dF, B', F, -true, true)
            mul!(dF, H, β, true, true)
            # dc = dot(_β, F) + 0.5*outer(F' * _σ) + 0.5*sum(H .* _a)
            dc[1] = 0.5*tr(tmat)
            mul!(tvec', F', a)
            mul!(dc, tvec', F, 0.5, true)
            mul!(dc, β', F, true, true)
            # ---
        end

        D = DiffusionDefinition.dimension(P).process
        el = choices.eltype

        problem_setup = (
            f = HFc_update!,
            HFcT = HFcContainer{el}(D),
            interval = (tt[end], tt[1]),
            buffer = HFcBuffer{el}(D),
        )
        update_HFc!(problem_setup.HFcT, xT_plus, obs)

        prob = ODEProblem(problem_setup...)
        saved_values = SavedValues(Float64, Tuple{Matrix{el},Vector{el},el})
        callback = SavingCallback(
            (u,t,integrator)->(u.H, u.F, u.c[1]),
            saved_values;
            saveat=reverse(tt),
            tdir=-1,
            save_everystep=false, # to prevent wasting memory allocations
        )
        sol = solve(prob, choices.solver, callback=callback)
        HFc0 = sol.u[end]
        Tsv, Tps = typeof(saved_values), typeof(problem_setup)
        Tcb, Ts, T = typeof(callback), typeof(choices.solver), typeof(HFc0)
        new{:inplace,Tsv,Tps,Tcb,Ts,T,Nothing}(
            saved_values,
            problem_setup,
            callback,
            choices.solver,
            HFc0,
            nothing
        )
    end

    function HFcSolver(
            ::Val{:outofplace},
            tt,
            xT_plus,
            P,
            obs,
            choices
        )
        access = Val{DiffusionDefinition.dimension(P).process}()
        function HFc_update(u, p, t)
            H, F, c = static_accessor_HFc(u, access)
            _B, _β, _σ, _a = DD.B(t, P), DD.β(t, P), DD.σ(t, P), DD.a(t, P)

            dH = - (_B' * H) - (H * _B) + outer(H * _σ)
            dF = - (_B' * F) + (H * _a * F) + (H * _β)
            dc = dot(_β, F) + 0.5*outer(F' * _σ) + 0.5*sum(H .* _a)
            vcat(SVector(dH), dF, SVector(dc))
        end

        problem_setup = (
            f = HFc_update,
            HFcT = update_HFc(xT_plus, obs, access),
            interval = (tt[end], tt[1]),
        )
        el = choices.eltype
        TH, TF, Tc = prepare_static_saving_types(Val{:hfc}(), access, el)
        prob = ODEProblem(problem_setup...)
        saved_values = SavedValues(Float64, Tuple{TH,TF,Tc})
        callback = SavingCallback(
            (u,t,integrator) -> static_accessor_HFc(u, access),
            saved_values;
            saveat=reverse(tt),
            tdir=-1,
            save_everystep=false,
        )
        sol = solve(prob, choices.solver, callback=callback)
        HFc0 = MVector(sol.u[end])
        Tsv, Tps = typeof(saved_values), typeof(problem_setup)
        Tcb, Ts, T = typeof(callback), typeof(choices.solver), typeof(HFc0)
        Ta = typeof(access)
        new{:outofplace,Tsv,Tps,Tcb,Ts,T,Ta}(
            saved_values,
            problem_setup,
            callback,
            choices.solver,
            HFc0,
            deepcopy(access),
        )
    end
end

"""
    HFc0(s::HFcSolver)

Return the data containing H,F,c terms for the time 0+
"""
HFc0(s::HFcSolver) = s.HFc0

"""
    update_HFc!(u_T, u_Tplus, obs, access)

Update equations for H,F,c at the times of observations. Save the data into
`u_T`.
"""
function update_HFc!(u_T, u_Tplus, obs)
    L, Λ, v, μ = obs.L, obs.Λ, obs.obs, obs.μ
    m, d = size(L)
    u_T.H .= u_Tplus.H + L'*Λ*L
    u_T.F .= u_Tplus.F + L*Λ*v
    u_T.c .= (u_Tplus.c .+ 0.5*( m*log(2π) + log(det(obs.Σ)) + (v-μ)'*Λ*(v-μ) ))
end

"""
    update_HFc(u_Tplus, obs, access)

Update equations for H,F,c at the times of observations.
"""
function update_HFc(u_Tplus, obs, access)
    L, Λ, v, μ = obs.L, obs.Λ, obs.obs, obs.μ
    m, d = size(L)
    H, F, c = static_accessor_HFc(u_Tplus, access)
    dH = L'*Λ*L
    dF = L*Λ*v
    dc = 0.5*( m*log(2π) + log(det(obs.Σ)) + (v-μ)'*Λ*(v-μ) )
    vcat(SVector(H + dH), (F + dF), SVector(c+dc))
end

"""
    prepare_static_saving_types(::Val{:hfc}, ::Val{D}, el) where D

Define data-types for H,F,c computed by out-of-place solver that are to be saved
internally.
"""
function prepare_static_saving_types(::Val{:hfc}, ::Val{D}, el) where D
    SMatrix{D,D,el,D*D}, SVector{D,el}, el
end

"""
    H(s::HFcSolver, i::Integer)

Return saved matrix H[i] (with H[1] indicating H at time 0+ and H[end]
indicating H at time T).
"""
H(s::HFcSolver, i::Integer) = s.saved_values.saveval[end-i+1][1]

"""
    F(s::HFcSolver, i::Integer)

Return saved vector F[i] (with F[1] indicating F at time 0+ and F[end]
indicating F at time T).
"""
F(s::HFcSolver, i::Integer) = s.saved_values.saveval[end-i+1][2]

"""
    c(s::HFcSolver, i::Integer)

Return saved scalar c[i] (with c[1] indicating c at time 0+ and c[end]
indicating c at time T).
"""
c(s::HFcSolver, i::Integer) = s.saved_values.saveval[end-i+1][3]


function recompute_guiding_term(
        s::HFcSolver{:inplace},
        P,
        obs,
        xT_plus
    )
    update_HFc!(s.problem_setup.HFcT, xT_plus, obs)
    prob = ODEProblem(s.problem_setup...)
    sol = solve(prob, s.solver, callback=s.callback)
    s.HFc0 .= sol.u[end]
end

function recompute_guiding_term(
        s::HFcSolver{:outofplace},
        P,
        obs,
        xT_plus
    )
    prob = ODEProblem(
        s.problem_setup.f,
        update_HFc(xT_plus, obs, s.access),
        s.problem_setup.interval
    )
    sol = solve(prob, s.solver, callback=s.callback)
    s.HFc0 .= sol.u[end]
end


∇logρ(i, x, P::HFcSolver{:outofplace}) = F(P, i) - H(P, i)*x

function ∇logρ!(buffer, i, x, P::HFcSolver{:inplace})
    mul!(buffer.∇logρ, H(P, i), x, -true, false)
    _F = F(P, i)
    for j in 1:length(buffer.∇logρ)
        buffer.∇logρ[j] += _F[j]
    end
end
