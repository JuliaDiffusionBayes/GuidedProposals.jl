function solve_and_ll!(XX, WW, P, y1)
    solve_and_ll!(XX, WW, P, P.guiding_term_solver, y1)
end

# simulate guided proposals and compute the likelihood at the same time
function solve_and_ll!(
        XX::Trajectory{T,Vector{KX}},
        WW::Trajectory{T,Vector{KW}},
        P::GuidProp,
        ::AbstractGuidingTermSolver{:outofplace},
        y1::KX,
    ) where {KX,KW,T}
    yy, ww, tt = XX.x, WW.x, XX.t
    N = length(XX)
    ll = 0.0

    yy[1] = y1
    for i in 1:(N-1)
        x = yy[i]
        s = tt[i]
        dt = tt[i+1] - tt[i]
        dW = ww[i+1] - ww[i]

        r_i = ∇logρ(i, x, P)
        b_i = DD.b(s, x, P.P_target)
        btil_i = DD.b(s, x, P.P_aux)

        σ_i = DD.σ(s, x, P.P_target)
        a_i = σ_i*σ_i'

        ll += dot(b_i-btil_i, r_i) * dt

        if !DD.constdiff(P)
            H_i = H(i, x, P)
            atil_i = DD.a(s, x, P.P_aux)
            ll += 0.5*tr( (a_i - atil_i)*(r_i*r_i'-H_i') ) * dt
        end

        yy[i+1] = x + (a_i*r_i + b_i)*dt + σ_i*dW

        DD.bound_satisfied(P, yy[i+1]) || return false, -Inf
    end
    true, ll
end

#NOTE worry about it later
function solve_and_ll!(
        XX::Trajectory{T,Vector{Vector{K}}},
        WW::Trajectory{T,Vector{Vector{K}}},
        P::GuidProp,
        ::AbstractGuidingTermSolver{:inplace},
        y1::Vector{K},
    ) where {K,T}
end


function Trajectories.trajectory(
        P::GuidProp,
        v::Type=DD.default_type(P),
        w::Type=DD.default_wiener_type(P)
    )
    trajectory(time(P), P.P_target, v, w)
end

function Trajectories.trajectory(
        PP::AbstractArray{<:GuidProp},
        v::Type=default_type(P),
        w::Type=default_wiener_type(P)
    )
    (
        process = [
            DD._process_traj(time(P), P.P_target, v) for P in PP
        ],
        wiener = [
            DD._wiener_traj(time(P), P.P_target, w) for P in PP
        ],
    )
end

#===============================================================================
                Simple sampling over a single interval
===============================================================================#

function Base.rand(P::GuidProp, y1=zero(P); f=DD._DEFAULT_F)
    rand(Random.GLOBAL_RNG, P, y1, DD.ismutable(y1); f=f)
end

function Base.rand(
        rng::Random.AbstractRNG,
        P::GuidProp,
        y1::K,
        v::Val{false};
        f=DD._DEFAULT_F
    ) where K
    w0 = (
        DD.default_wiener_type(P) <: Number ?
        zero(eltype(K)) :
        zero(similar_type(K, Size(DD.dim_wiener(P.P_target))))
    )
    Wnr = Wiener()
    X, W = trajectory(P, typeof(y1), typeof(w0))
    success, f_accum = false, nothing
    while !success
        rand!(rng, Wnr, W, w0)
        success, f_accum = DD.solve!(X, W, P, y1; f=f)
    end
    typeof(f) != typeof(DD._DEFAULT_F) && return X, W, Wnr, f_accum
    X, W, Wnr
end

#===============================================================================
                in-place sampling over a single interval
===============================================================================#

function Random.rand!(P::GuidProp, X, W, y1=zero(P); f=DD._DEFAULT_F, Wnr=Wiener())
    rand!(Random.GLOBAL_RNG, P, X, W, y1, DD.ismutable(y1); f=f, Wnr=Wnr)
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X,
        W,
        y1::K,
        v::Val{false};
        f=DD._DEFAULT_F,
        Wnr=Wiener(),
    ) where K
    rand!(rng, Wnr, W)
    DD.solve!(X, W, P, y1; f=f)
end


function Random.rand!(P::GuidProp, X°, W°, W, ρ, y1=zero(P); f=DD._DEFAULT_F, Wnr=Wiener())
    rand!(Random.GLOBAL_RNG, P, X°, W°, W, ρ, y1, DD.ismutable(y1); f=f, Wnr=Wnr)
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X°,
        W°,
        W,
        ρ,
        y1::K,
        v::Val{false};
        f=DD._DEFAULT_F,
        Wnr=Wiener(),
    ) where K
    rand!(rng, Wnr, W°)
    crank_nicolson!(W°.x, W.x, ρ)
    DD.solve!(X°, W°, P, y1; f=f)
end


function Random.rand!(P::GuidProp, X, W, v::Val{:ll}, y1=zero(P); Wnr=Wiener())
    rand!(Random.GLOBAL_RNG, P, X, W, v, y1, DD.ismutable(y1); Wnr=Wnr)
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X,
        W,
        ::Val{:ll},
        y1::K,
        ::Val{false};
        Wnr=Wiener(),
    ) where K
    rand!(rng, Wnr, W)
    success, ll = solve_and_ll!(X, W, P, y1)
end


function Random.rand!(P::GuidProp, X°, W°, W, ρ, v::Val{:ll}, y1=zero(P); Wnr=Wiener())
    rand!(Random.GLOBAL_RNG, P, X°, W°, W, ρ, v, y1, DD.ismutable(y1); Wnr=Wnr)
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X°,
        W°,
        W,
        ρ,
        ::Val{:ll},
        y1::K,
        ::Val{false};
        Wnr=Wiener(),
    ) where K
    rand!(rng, Wnr, W°)
    crank_nicolson!(W°.x, W.x, ρ)
    solve_and_ll!(X°, W°, P, y1)
end


#===============================================================================
                Simple sampling over multiple intervals
===============================================================================#



function Base.rand(
        P::AbstractArray{<:GuidProp},
        y1=zero(P[1]);
        f=DD._DEFAULT_F
    )
    rand(Random.GLOBAL_RNG, P, y1; f=f)
end

function Base.rand(
        rng::Random.AbstractRNG,
        PP::AbstractArray{<:GuidProp},
        y1::K;
        f=DD._DEFAULT_F
    ) where K
    results = map(1:length(PP)) do i
        result = rand(rng, PP[i], y1, DD.ismutable(y1); f=f)
        y1 = result[1].x[end]
        result
    end
    XX = map(r->r[1], results)
    WW = map(r->r[2], results)
    Wnr = results[3][1]
    typeof(f) != typeof(DD._DEFAULT_F) && return XX, WW, Wnr, map(r->r[4], results)
    XX, WW, Wnr
end

#===============================================================================
                in-place sampling over multiple intervals
===============================================================================#
#=
function Random.rand!(P::AbstractArray{<:GuidProp}, X, W, y1=zero(P); f=DD._DEFAULT_F, Wnr=Wiener())
    rand!(Random.GLOBAL_RNG, P, X, W, y1, DD.ismutable(y1); f=f, Wnr=Wnr)
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::AbstractArray{<:GuidProp},
        X,
        W,
        y1::K,
        v::Val{false};
        f=DD._DEFAULT_F,
        Wnr=Wiener(),
    ) where K
    rand!(rng, Wnr, W)
    DD.solve!(X, W, P, y1; f=f)
end
=#
