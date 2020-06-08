#===============================================================================

    Routines for computing the log-likelihood functions and solve!'ing
    the path from the Wiener path and computing the log-likelihood at
    the same time.

===============================================================================#

@doc raw"""
    loglikhd([::IntegrationRule=::LeftRule], P::GuidProp, X::Trajectory; skip=0)

Compute path contribution to the log-likelihood function, i.e.:
```math
\int_0^T G(t, X_t) dt,
```
where
```math
G(t,x):=\left[
    (b-\tilde{b})^T\tilde{r}
    + 0.5 tr\left\{
        (a-\tilde{a})(\tilde{r}\tilde{r}^T-H)
    \right\}
\right](t,x),
```
and
$\tilde{r}(t,x):=\nabla\log\tilde{\rho}(t,x)$.

    loglikhd(
        [::IntegrationRule=::LeftRule],
        PP::AbstractArray{<:GuidProp}, XX::AbstractArray{<:Trajectory};
        skip=0
    )
Compute path contribution to the log-likelihood function for a sequence of
segments.
"""
function loglikhd end

function loglikhd(P::GuidProp, X::Trajectory; skip=0)
    loglikhd(LeftRule(), P, X, P.guiding_term_solver; skip=skip)
end

function loglikhd(ir::IntegrationRule, P::GuidProp, X::Trajectory; skip=0)
    loglikhd(ir, P, X, P.guiding_term_solver; skip=skip)
end

function loglikhd(
        ::LeftRule,
        P::GuidProp,
        X::Trajectory,
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
            ll += 0.5 * tr( (a_i - atil_i)*(r_i*r_' - H_i) ) * dt
        end
    end
    ll
end

# NOTE worry about this later...
function loglikhd(
        ::LeftRule,
        P::GuidProp,
        X::Trajectory,
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

function loglikhd(
        PP::AbstractArray{<:GuidProp},
        XX::AbstractArray{<:Trajectory};
        skip=0
    )
    loglikhd(LeftRule(), PP, XX; skip=skip)
end

function loglikhd(
        ir::IntegrationRule,
        PP::AbstractArray{<:GuidProp},
        XX::AbstractArray{<:Trajectory};
        skip=0
    )
    ll_tot = loglikhd_obs(PP[1], XX[1].x[1])
    for i in eachindex(PP)
        ll_tot += loglikhd(ir, PP[i], XX[i], PP[i].guiding_term_solver; skip=skip)
    end
    ll_tot
end

@doc raw"""
    loglikhd_obs(P::GuidProp, x0)

Compute the contribution of end-points to the log-likelihood function, i.e.:
$\log\rho(t,x)$.
"""
loglikhd_obs(P::GuidProp, x0) = loglikhd_obs(P, x0, P.guiding_term_solver)

function loglikhd_obs(P::GuidProp, x0, ::AbstractGuidingTermSolver{:outofplace})
    - 0.5 * ( x0'*H(P, 1)*x0 - 2.0*dot(F(P, 1), x0) ) - c(P, 1)
end

# worry about it later
function loglikhd_obs(P::GuidProp, x0, ::AbstractGuidingTermSolver{:inplace})
    - 0.5 * ( x0'*H(P, 1)*x0 - 2.0*dot(F(P, 1), x₀) ) - c(P, 1)
end

#===============================================================================
            solve! with simultaneous computation of log-likelihood
===============================================================================#
"""
    solve_and_ll!(X, W, P, y1)

Compute the trajectory under the law `P` for a given Wiener noise `W` and
a starting point `y1`. Store the trajectory in `XX`. Compute the log-likelihood
(path contribution) along the way. Return `success_flag` and log-likelihood.
`success_flag` is set to false only if sampling was prematurely halted due to
`XX` violating assumptions about state space.
"""
function solve_and_ll!(X, W, P::GuidProp, y1; skip=0)
    solve_and_ll!(X, W, P, P.guiding_term_solver, y1; skip=skip)
end

function solve_and_ll!(XX, WW, PP::AbstractArray{<:GuidProp}, y1; skip=0)
    ll_tot = loglikhd_obs(PP[1], y1)
    for i in eachindex(PP)
        success, ll = solve_and_ll!(XX[i], WW[i], PP[i], PP[i].guiding_term_solver, y1; skip=skip)
        success || return false, ll
        ll_tot += ll
        y1 = XX[i].x[end]
    end
    true, ll_tot
end

function solve_and_ll!(
        XX,
        WW,
        P::GuidProp,
        ::AbstractGuidingTermSolver{:outofplace},
        y1;
        skip=0
    )
    yy, ww, tt = XX.x, WW.x, XX.t
    N = length(XX)

    yy[1] = DD.value(y1)
    x = y1
    ll = 0.0
    for i in 1:(N-1)
        add_to_ll = (i < N-skip)
        s = tt[i]
        dt = tt[i+1] - tt[i]
        dW = ww[i+1] - ww[i]

        r_i = ∇logρ(i, x, P)
        b_i = DD.b(s, x, P.P_target)
        btil_i = DD.b(s, x, P.P_aux)

        σ_i = DD.σ(s, x, P.P_target)
        a_i = σ_i*σ_i'

        add_to_ll && (ll += dot(b_i-btil_i, r_i) * dt)

        if !DD.constdiff(P)
            H_i = H(i, x, P)
            atil_i = DD.a(s, x, P.P_aux)
            add_to_ll && (ll += 0.5*tr( (a_i - atil_i)*(r_i*r_i'-H_i') ) * dt)
        end

        x = x + (a_i*r_i + b_i)*dt + σ_i*dW

        yy[i+1] = DD.value(x)

        DD.bound_satisfied(P, yy[i+1]) || return false, -Inf
    end
    true, ll
end

#NOTE worry about it later
function solve_and_ll!(
        XX,
        WW,
        P::GuidProp,
        ::AbstractGuidingTermSolver{:inplace},
        y1;
        skip=0
    )
end
