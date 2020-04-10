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
struct HFcSolver{Tmode,Tsv,Ti,TT,T0,Ta} <: AbstractGuidingTermSolver{Tmode}
    saved_values::Tsv
    integrator::Ti
    HFcT::TT
    HFc0::T0
    access::Ta

    function HFcSolver(
            ::Val{:inplace},
            tt,
            xT_plus,
            P,
            obs,
            choices,
        )
        D = DiffusionDefinition.dimension(P).process
        el = choices.eltype

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
            mul!(dH, tmat, H')
            mul!(dH, H, B, -true, true)
            mul!(dH, B', H, -true, true)
            # dF = - (_B' * F) + (H * a * F) + (H * _β)
            mul!(dF, tmat, F)
            mul!(dF, B', F, -true, true)
            mul!(dF, H, β, true, true)
            # dc = dot(_β, F) + 0.5*outer(F' * _σ) - 0.5*sum(H .* _a)
            # apparently, according to @time, a faster way to compute a trace,
            # with no allocations
            dc[1] = 0.0
            for i in 1:D
                dc[1] -= tmat[i,i]
            end
            dc[1] *= 0.5
            #@time dc[1] = -0.5*tr(tmat)
            mul!(tvec, a', F)
            mul!(dc, tvec', F, 0.5, true)
            mul!(dc, β', F, true, true)
            # ---
        end

        HFcT = HFcContainer{el}(D)
        update_HFc!(HFcT, xT_plus, obs)

        prob = ODEProblem{true}(
            HFc_update!,
            HFcT,
            (tt[end], tt[1]),
            HFcBuffer{el}(D)
        )
        saved_values = SavedValues(Float64, Tuple{Matrix{el},Vector{el},el})
        callback = SavingCallback(
            # NOTE this might need changing, even though it seems to be working
            # correctly; from
            # https://docs.sciml.ai/latest/features/callback_library/ :
            # "save_func(u, t, integrator) returns the quantities which shall be
            # saved. Note that this should allocate the output (not as a view to
            # u)". Currently u.H & u.F provide views!
            (u,t,integrator)->(u.H, u.F, u.c[1]),
            saved_values;
            saveat=reverse(tt),
            tdir=-1
        )
        integrator = init(
            prob,
            choices.solver;
            callback=callback,
            save_everystep=false, # to prevent wasting memory allocations
        )
        sol = solve!(integrator)
        HFc0 = sol.u[end]
        Tsv, Ti = typeof(saved_values), typeof(integrator), typeof(HFc0)
        TT, T0 = typeof(HFcT), typeof(HFc0)
        new{:inplace,Tsv,Ti,TT,T0,Nothing}(
            saved_values,
            integrator,
            HFcT,
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
            dc = dot(_β, F) + 0.5*outer(F' * _σ) - 0.5*tr(H*_a)
            vcat(SVector(dH), dF, SVector(dc))
        end

        el = choices.eltype
        TH, TF, Tc = prepare_static_saving_types(Val{:hfc}(), access, el)
        prob = ODEProblem{false}(
            HFc_update,
            update_HFc(xT_plus, obs, access),
            (tt[end], tt[1])
        )
        saved_values = SavedValues(Float64, Tuple{TH,TF,Tc})
        callback = SavingCallback(
            (u,t,integrator) -> static_accessor_HFc(u, access),
            saved_values;
            saveat=reverse(tt),
            tdir=-1
        )
        integrator = init(
            prob,
            choices.solver,
            callback=callback,
            save_everystep=false, # to prevent wasting memory allocations
        )
        sol = solve!(integrator)
        HFc0 = MVector(sol.u[end])
        Tsv, Ti, T0 = typeof(saved_values), typeof(integrator), typeof(HFc0)
        Ta = typeof(access)
        new{:outofplace,Tsv,Ti,Nothing,T0,Ta}(
            saved_values,
            integrator,
            nothing,
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
    L, Λ, Σ, v, μ = DOS.L(obs), DOS.Λ(obs), DOS.Σ(obs), DOS.ν(obs), DOS.μ(obs)
    m, d = size(L)
    u_T.H .= u_Tplus.H + L'*Λ*L
    u_T.F .= u_Tplus.F + L*Λ*v
    u_T.c .= (u_Tplus.c .+ 0.5*( m*log(2π) + log(det(Σ)) + (v-μ)'*Λ*(v-μ) ))
end

"""
    update_HFc(u_Tplus, obs, access)

Update equations for H,F,c at the times of observations.
"""
function update_HFc(u_Tplus, obs, access)
    L, Λ, Σ, v, μ = DOS.L(obs), DOS.Λ(obs), DOS.Σ(obs), DOS.ν(obs), DOS.μ(obs)
    m, _ = size(L)
    H, F, c = static_accessor_HFc(u_Tplus, access)
    dH = L'*Λ*L
    dF = L*Λ*v
    dc = 0.5*( m*log(2π) + log(det(Σ)) + (v-μ)'*Λ*(v-μ) )
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

function recompute_guiding_term!(
        s::HFcSolver{:inplace},
        P,
        obs,
        xT_plus
    )
    update_HFc!(s.HFcT, xT_plus, obs)
    reinit!(s.integrator, s.HFcT)
    sol = solve!(s.integrator)
    s.HFc0 .= sol.u[end]
end

function recompute_guiding_term!(
        s::HFcSolver{:outofplace},
        P,
        obs,
        xT_plus
    )
    reinit!(s.integrator, update_HFc(xT_plus, obs, s.access))
    sol = solve!(s.integrator)
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
