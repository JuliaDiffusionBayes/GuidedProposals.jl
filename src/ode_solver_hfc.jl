#==============================================================================#
#
#       ODE solver for H,F,c terms; relies on DifferentialEquations.jl
#
#==============================================================================#
"""
    struct HFcSolver{Tp,Tcb,T}

Struct for solving a (H,F,c)-system of ODEs.

        HFcSolver(
            ::Val{:inplace},
            tt,
            xT_plus,
            P,
            solver_type,
            save_as_type,
        )
    Constructor for an ODE solver with in-place operations. Pre-allocates space
    and solves once a (H,F,c)-system of ODEs backward in time on the interval
    `(tt[1], tt[end])`, with a terminal condition computed from `xT_plus`. `P`
    is the auxiliary diffusion, and `solver_type` indicates the algorithm for
    solving ODEs. `save_as_type` gives datatypes that H,F,c are to be saved as
    and `tt` gives a grid of time-points at which they need to be saved.

        HFcSolver(
            ::Val{:outofplace},
            tt,
            xT_plus,
            P,
            solver_type,
            save_as_type,
        )
    Constructor for an ODE solver with out-of-place operations.
    NOTE: currently not implemented
"""
struct HFcSolver{Tp,Tcb,T}
    prob::Tp
    callback::Tcb
    HFcT::T
    HFc0::T

    function HFcSolver(
            ::Val{:inplace},
            tt,
            xT_plus,
            P,
            obs,
            solver_type,
            save_as_type,
        )
        access = Val{dimension(P).process}()
        HFcT = copy(xT_plus)
        update_HFc!(HFcT, xT_plus, obs, access)

        function HFc_update!(du, u, p, t)
            # in-place comp. of auxiliary proc.; stored in du
            B!(du, t, P), β!(du, t, P), σ!(du, t, P), a!(du, t, P)

            # shorthand names for views, hopefully optimised
            # ---
            # current state
            H, F, c = _H(u, access), _F(u, access), _c(u, access)
            B, β, σ = _B(du, access), _β(du, access), _σ(du, access)
            a = _a(du, access)
            # increments (to-be-computed here)
            dH, dF, dc = _H(du, access), _F(du, access), _c(du, access)
            # temporary variables
            tH = tempH(du, access)
            # ---

            # ODEs
            # ---
            # dH = - (_B' * H) - (H * _B) + outer(H * σ)
            mul!(tH, H, a)
            mul!(dH, tH, H')
            mul!(dH, H, B, -true, true)
            mul!(dH, B', H, -true, true)
            # dF = - (_B' * F) + (H * a * F) + (H * _β)
            mul!(dF, tH, F)
            mul!(dF, B', F, -true, true)
            mul!(dF, H, β, true, true)
            # dc = dot(_β, F) + 0.5*outer(F' * _σ) + 0.5*sum(H .* _a)
            dc[1] = 0.5*tr(tH)
            mul!(tc, F', a)
            mul!(dc, tc, F, 0.5, true)
            mul!(dc, β', F, true, true)
            # ---
        end

        TH, TF, Tc = save_as_type
        prob = ODEProblem(HFc_update!, HFcT, (tt[end], tt[1]))
        saved_values = SavedValues(Float64, Tuple{TH,TF,Tc})
        callback = SavingCallback(
            (u,t,integrator)->(
                _H(u,access),
                _F(u,access),
                _c(u,access),
            ),
            saved_values;
            saveat=reverse(tt),
            tdir=-1,
        )
        sol = solve(prob, solver_type, callback=callback)
        HFc0 = sol.u[end]
        Tp, Tcb, T = typeof(prob), typeof(callback), typeof(HFcT)
        new{Tp,Tcb,T}(prob, callback, HFc0)
    end

    #TODO let's worry about this later
    #=
    function HFcSolver(::Any, (t0, T), xT, dim_of_process, P, solver_type)
        access = Val{dim_of_process}()

        function HFc_update!(du, u, p, t)
            H, F, c = _H(u, access), _F(u, access), _c(u, access)
            _B, _β, _σ, _a = B!(u, t, P), β!(u, t, P), σ!(u, t, P), a!(u, t, P)

            _H(du, access) .= - (_B' * H) - (H * _B) + outer(H * _σ)
            _F(du, access) .= - (_B' * F) + (H * a * F) + (H * _β)
            _c(du, access) .= dot(_β, F) + 0.5*outer(F' * _σ) + 0.5*sum(H .* _a)
        end

        prob = ODEProblem(HFc_update!, xT, (T, t0))
        saved_values = SavedValues(Float64, Tuple{TH,TF,Tc})
        callback = SavingCallback(
            (u,t,integrator)->(
                TH( _H(u,access) ),
                TF( _F(u,access) ),
                Tc( _c(u,access) ),
            ),
            saved_values
        )
        Tcb = typeof(callback)
        solve(prob, solver_type, callback=callback)
        new{TH,TF,Tc,Tcb}(saved_values, callback)
    end
    =#
end

"""
    HFc0(s::HFcSolver)

Return the data containing H,F,c terms for the time 0+
"""
HFc0(s::HFcSolver) = s.HFc0

"""
    update_HFc!(u_T, u_Tplus, obs, access)

Update equations for H,F,c at the times of observations.
"""
function update_HFc!(u_T, u_Tplus, obs, access)
    L, Λ, v, μ = obs.L, obs.Λ, obs.obs, obs.μ
    m, d = size(L)
    _H(u_T, access) .= _H(u_Tplus, access) + reshape(L'*Λ*L, d*d)
    _F(u_T, access) .= _F(u_Tplus, access) + reshape(L*Λ*v, d)
    _c(u_T, access) .= (
        _c(u_Tplus, access)
        + 0.5*( m*log(2π) + log(det(obs.Σ)) + (v-μ)'*Λ*(v-μ) )
    )
end
