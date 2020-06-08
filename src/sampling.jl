#===============================================================================
                Extensions of Trajectories for GuidProp
===============================================================================#
function Trajectories.trajectory(
        P::GuidProp,
        v::Type=DD.default_type(P),
        w::Type=DD.default_wiener_type(P)
    )
    trajectory(time(P), P.P_target, v, w)
end

function Trajectories.trajectory(
        PP::AbstractArray{<:GuidProp},
        v::Type=DD.default_type(PP[1]),
        w::Type=DD.default_wiener_type(PP[1])
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
                preconditioned Crank-Nicolson scheme
===============================================================================#
const Mutable = Array

function crank_nicolson!(y°, y, ρ) # For immutable types
    λ = sqrt(1-ρ^2)
    for i in 1:length(y)
        y°[i] = λ*y°[i] + ρ*y[i]
    end
end

function crank_nicolson!(y°::Vector{T}, y, ρ) where {T<:Mutable} #NOTE GPUs will need to be treated separately
    λ = sqrt(1-ρ^2)
    for i in 1:length(y)
        mul!(y°[i], y[i], true, ρ, λ)
    end
end

#===============================================================================
                Simple sampling over a single interval
===============================================================================#
"""
    Base.rand(
        [rng::Random.AbstractRNG], P::GuidProp, y1=zero(P); f=DD.__DEFAULT_F
    )

Sample a trajectory of a guided proposal `P` started from `y1`. Initialize
containers in the background and compute the functional `f` at the time of
sampling.
"""
function Base.rand(P::GuidProp, y1=zero(P); f=DD.__DEFAULT_F)
    rand(Random.GLOBAL_RNG, P, y1, DD.ismutable(y1); f=f)
end

function Base.rand(
        rng::Random.AbstractRNG, P::GuidProp, y1=zero(P); f=DD.__DEFAULT_F
    )
    rand(rng, P, y1, DD.ismutable(y1); f=f)
end

function Base.rand(
        rng::Random.AbstractRNG,
        P::GuidProp, y1::K, v::Val{false};
        f=DD.__DEFAULT_F
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
    typeof(f) != DD._DEFAULT_F && return X, W, Wnr, f_accum
    X, W, Wnr
end

function Base.rand(
        rng::Random.AbstractRNG,
        P::GuidProp, y1::K, v::Val{true};
        f=DD.__DEFAULT_F
    ) where K
    error("in-place not implemented")
end

#===============================================================================
                in-place sampling over a single interval
===============================================================================#

#=---- Vanilla ----=#
"""
    Random.rand!(
        [rng::Random.AbstractRNG],
        P::GuidProp,
        X, W, y1=zero(P);
        f=DD.__DEFAULT_F, Wnr=Wiener()
    )

Sample a trajectory of a guided proposal `P` started from `y1`. Use containers
`X` and `W` to save the results. Compute the functional `f` at the time of
sampling.
"""
function Random.rand!(
        P::GuidProp,
        X, W, y1=zero(P);
        f=DD.__DEFAULT_F, Wnr=Wiener()
    )
    rand!(
        Random.GLOBAL_RNG,
        P, X, W, y1, DD.ismutable(y1);
        f=f, Wnr=Wnr
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X, W, y1=zero(P);
        f=DD.__DEFAULT_F, Wnr=Wiener()
    )
    rand!(
        rng,
        P, X, W, y1, DD.ismutable(y1);
        f=f, Wnr=Wnr
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X, W, y1::K, v::Val{false};
        f=DD.__DEFAULT_F, Wnr=Wiener(),
    ) where K
    rand!(rng, Wnr, W)
    DD.solve!(X, W, P, y1; f=f)
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X, W, y1::K, v::Val{true};
        f=DD.__DEFAULT_F, Wnr=Wiener(),
    ) where K
    error("in-place not implemented")
end

#=---- with Crank-Nicolson scheme ----=#
"""
    Random.rand!(
        [rng::Random.AbstractRNG],
        P::GuidProp,
        X°, W°, W, ρ, y1=zero(P);
        f=DD.__DEFAULT_F, Wnr=Wiener()
    )

Sample a trajectory of a guided proposal `P` started from `y1`. Use containers
`X°` and `W°` to save the results. Use a preconditioned Crank-Nicolson scheme
with memory parameter `ρ` and a previously sampled Wiener noise `W`. Compute the
functional `f` at the time of sampling.
"""
function Random.rand!(
        P::GuidProp,
        X°, W°, W, ρ, y1=zero(P);
        f=DD.__DEFAULT_F, Wnr=Wiener()
    )
    rand!(
        Random.GLOBAL_RNG,
        P, X°, W°, W, ρ, y1, DD.ismutable(y1);
        f=f, Wnr=Wnr
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X°, W°, W, ρ, y1=zero(P);
        f=DD.__DEFAULT_F, Wnr=Wiener()
    )
    rand!(
        rng,
        P, X°, W°, W, ρ, y1, DD.ismutable(y1);
        f=f, Wnr=Wnr
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X°, W°, W, ρ, y1::K, v::Val{false};
        f=DD.__DEFAULT_F, Wnr=Wiener(),
    ) where K
    rand!(rng, Wnr, W°)
    crank_nicolson!(W°.x, W.x, ρ)
    DD.solve!(X°, W°, P, y1; f=f)
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X°, W°, W, ρ, y1::K, v::Val{true};
        f=DD.__DEFAULT_F, Wnr=Wiener(),
    ) where K
    error("in-place not implemented")
end

#=---- with log-likelihood ----=#
"""
    Random.rand!(
        [rng::Random.AbstractRNG]
        P::GuidProp,
        X, W, v::Val{:ll}, y1=zero(P);
        Wnr=Wiener()
    )

Sample a trajectory of a guided proposal `P` started from `y1`. Use containers
`X` and `W` to save the results. Compute log-likelihood (only path contribution)
along the way.
"""
function Random.rand!(
        P::GuidProp,
        X, W, v::Val{:ll}, y1=zero(P);
        Wnr=Wiener(), skip=0
    )
    rand!(
        Random.GLOBAL_RNG,
        P, X, W, v, y1, DD.ismutable(y1);
        Wnr=Wnr, skip=skip
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X, W, v::Val{:ll}, y1=zero(P);
        Wnr=Wiener(), skip=0
    )
    rand!(
        rng,
        P, X, W, v, y1, DD.ismutable(y1);
        Wnr=Wnr, skip=skip
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X, W, ::Val{:ll}, y1::K, ::Val{false};
        Wnr=Wiener(), skip=0
    ) where K
    rand!(rng, Wnr, W)
    solve_and_ll!(X, W, P, y1; skip=skip)
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X, W, ::Val{:ll}, y1::K, ::Val{true};
        Wnr=Wiener(), skip=0
    ) where K
    error("in-place not implemented")
end

#=---- with log-likelihood and Crank-Nicolson scheme ----=#
"""
    Random.rand!(
        [rng::Random.AbstractRNG]
        P::GuidProp,
        X°, W°, W, ρ, v::Val{:ll}, y1=zero(P);
        Wnr=Wiener()
    )

Sample a trajectory of a guided proposal `P` started from `y1`. Use containers
`X°` and `W°` to save the results. Use a preconditioned Crank-Nicolson scheme
with memory parameter `ρ` and a previously sampled Wiener noise `W`. Compute
log-likelihood (only path contribution) along the way.
"""
function Random.rand!(
        P::GuidProp,
        X°, W°, W, ρ, v::Val{:ll}, y1=zero(P);
        Wnr=Wiener(), skip=0
    )
    rand!(
        Random.GLOBAL_RNG,
        P, X°, W°, W, ρ, v, y1, DD.ismutable(y1);
        Wnr=Wnr, skip=skip
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X°, W°, W, ρ, v::Val{:ll}, y1=zero(P);
        Wnr=Wiener(), skip=0
    )
    rand!(
        rng,
        P, X°, W°, W, ρ, v, y1, DD.ismutable(y1);
        Wnr=Wnr, skip=skip
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X°, W°, W, ρ, ::Val{:ll}, y1::K, ::Val{false};
        Wnr=Wiener(), skip=0
    ) where K
    rand!(rng, Wnr, W°)
    crank_nicolson!(W°.x, W.x, ρ)
    solve_and_ll!(X°, W°, P, y1; skip=skip)
end

function Random.rand!(
        rng::Random.AbstractRNG,
        P::GuidProp,
        X°, W°, W, ρ, ::Val{:ll}, y1::K, ::Val{true};
        Wnr=Wiener(),
    ) where K
    error("in-place not implemented")
end


#===============================================================================
                Simple sampling over multiple intervals
===============================================================================#
"""
    Base.rand(
        [rng::Random.AbstractRNG],
        PP::AbstractArray{<:GuidProp}, y1=zero(PP[1]); f=DD.__DEFAULT_F
    )

Sample a trajectory started from `y1`, defined for multiple guided proposals
`PP` that correspond to consecutive intervals. Initialize containers in the
background and compute the functionals `f` (one for each interval) at the time
of sampling.
"""
function Base.rand(
        PP::AbstractArray{<:GuidProp},
        y1=zero(PP[1]);
        f=DD.__DEFAULT_F
    )
    rand(Random.GLOBAL_RNG, PP, y1; f=f)
end

function Base.rand(
        rng::Random.AbstractRNG,
        PP::AbstractArray{<:GuidProp},
        y1::K;
        f=DD.__DEFAULT_F
    ) where K
    results = map(1:length(PP)) do i
        result = rand(rng, PP[i], y1, DD.ismutable(y1); f=f[i])
        y1 = result[1].x[end]
        result
    end
    XX = map(r->r[1], results)
    WW = map(r->r[2], results)
    Wnr = results[1][3]
    typeof(f) != DD._DEFAULT_F && return XX, WW, Wnr, map(r->r[4], results)
    XX, WW, Wnr
end

#===============================================================================
                out-of-place sampling over multiple intervals
===============================================================================#

#=---- Vanilla ----=#
"""
    Random.rand!(
        [rng::Random.AbstractRNG]
        PP::AbstractArray{<:GuidProp},
        XX, WW, y1=zero(PP[1]);
        f=DD.__DEFAULT_F, f_out=DD.__DEFAULT_F, Wnr=Wiener()
    )

Sample a trajectory started from `y1` over multiple intervals for guided
proposals `PP` that correspond to consecutive intervals. Use containers `XX` and
`WW` to save the results. Compute the functionals `f` (one for each interval) at
the time of sampling and store the results in `f_out`.
"""
function Random.rand!(
        PP::AbstractArray{<:GuidProp},
        XX, WW, y1=zero(PP[1]);
        f=DD.__DEFAULT_F, f_out=DD.__DEFAULT_F, Wnr=Wiener()
    )
    rand!(
        Random.GLOBAL_RNG,
        PP, XX, WW, y1, DD.ismutable(y1);
        f=f, f_out=f_out, Wnr=Wnr
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        PP::AbstractArray{<:GuidProp},
        XX, WW, y1=zero(PP[1]);
        f=DD.__DEFAULT_F, f_out=DD.__DEFAULT_F, Wnr=Wiener()
    )
    rand!(
        rng,
        PP, XX, WW, y1, DD.ismutable(y1);
        f=f, f_out=f_out, Wnr=Wnr
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        PP::AbstractArray{<:GuidProp},
        XX, WW, y1::K, v::Val{false};
        f=DD.__DEFAULT_F, f_out=DD.__DEFAULT_F, Wnr=Wiener(),
    ) where K
    for i in eachindex(PP)
        rand!(rng, Wnr, WW[i])
        success, f_out[i] = rand!(
            rng, PP[i], XX[i], WW[i], y1, v; f=f[i], Wnr=Wnr
        )
        success || return false
        y1 = XX[i].x[end]
    end
    true
end

function Random.rand!(
        rng::Random.AbstractRNG,
        PP::AbstractArray{<:GuidProp},
        XX, WW, y1::K, v::Val{true};
        f=DD.__DEFAULT_F, f_out=DD.__DEFAULT_F, Wnr=Wiener(),
    ) where K
    error("in-place not implemented")
end

#=---- with Crank-Nicolson scheme ----=#
"""
    Random.rand!(
        [rng::Random.AbstractRNG]
        PP::AbstractArray{<:GuidProp},
        XX°, WW°, WW, ρρ, y1=zero(PP[1]);
        f=DD.__DEFAULT_F, f_out=DD.__DEFAULT_F, Wnr=Wiener()
    )

Sample a trajectory started from `y1` over multiple intervals for guided
proposals `PP` that correspond to consecutive intervals. Use containers `XX°`
and `WW°` to save the results. Use a preconditioned Crank-Nicolson scheme
with memory parameters `ρρ` (one for each interval) and a previously sampled
Wiener noise `WW`. Compute the functionals `f` (one for each interval) at
the time of sampling and store the results in `f_out`.
"""
function Random.rand!(
        PP::AbstractArray{<:GuidProp},
        XX°, WW°, WW, ρρ, y1=zero(PP[1]);
        f=DD.__DEFAULT_F, f_out=DD.__DEFAULT_F, Wnr=Wiener()
    )
    rand!(
        Random.GLOBAL_RNG,
        PP, XX°, WW°, WW, ρρ, y1, DD.ismutable(y1);
        f=f, f_out=f_out, Wnr=Wnr
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        PP::AbstractArray{<:GuidProp},
        XX°, WW°, WW, ρρ, y1=zero(PP[1]);
        f=DD.__DEFAULT_F, f_out=DD.__DEFAULT_F, Wnr=Wiener()
    )
    rand!(
        rng,
        PP, XX°, WW°, WW, ρρ, y1, DD.ismutable(y1);
        f=f, f_out=f_out, Wnr=Wnr
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        PP::AbstractArray{<:GuidProp},
        XX°, WW°, WW, ρρ, y1::K, v::Val{false};
        f=DD.__DEFAULT_F, f_out=DD.__DEFAULT_F, Wnr=Wiener(),
    ) where K
    for i in eachindex(PP)
        success, f_out[i] = rand!(
            rng, PP[i], XX°[i], WW°[i], WW[i], ρρ[i], y1, v; f=f[i], Wnr=Wnr
        )
        success || return false
        y1 = XX°[i].x[end]
    end
    true
end

function Random.rand!(
        rng::Random.AbstractRNG,
        PP::AbstractArray{<:GuidProp},
        XX°, WW°, WW, ρρ, y1::K, v::Val{true};
        f=DD.__DEFAULT_F, f_out=DD.__DEFAULT_F, Wnr=Wiener(),
    ) where K
    error("in-place not implemented")
end


#=---- with log-likelihood ----=#
"""
    Random.rand!(
        [rng::Random.AbstractRNG]
        PP::AbstractArray{<:GuidProp},
        XX, WW, v::Val{:ll}, y1=zero(PP[1]);
        Wnr=Wiener()
    )

Sample a trajectory started from `y1` over multiple intervals for guided
proposals `PP` that correspond to consecutive intervals. Use containers `XX°`
and `WW°` to save the results. Compute log-likelihood (path contribution AND
end-points contribution) along the way.
"""
function Random.rand!(
        PP::AbstractArray{<:GuidProp},
        XX, WW, v::Val{:ll}, y1=zero(PP[1]);
        Wnr=Wiener(), skip=0
    )
    rand!(
        Random.GLOBAL_RNG,
        PP, XX, WW, v, y1, DD.ismutable(y1);
        Wnr=Wnr, skip=skip
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        PP::AbstractArray{<:GuidProp},
        XX, WW, v::Val{:ll}, y1=zero(PP[1]);
        Wnr=Wiener(), skip=0
    )
    rand!(
        rng,
        PP, XX, WW, v, y1, DD.ismutable(y1);
        Wnr=Wnr, skip=skip
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        PP::AbstractArray{<:GuidProp},
        XX, WW, v::Val{:ll}, y1::K, m::Val{false};
        Wnr=Wiener(), skip=0
    ) where K
    ll_tot = loglikhd_obs(PP[1], y1)
    for i in eachindex(PP)
        success, ll = rand!(
            rng, PP[i], XX[i], WW[i], v, y1, m; Wnr=Wnr, skip=skip
        )
        success || return false, ll
        ll_tot += ll
        y1 = XX[i].x[end]
    end
    true, ll_tot
end

function Random.rand!(
        rng::Random.AbstractRNG,
        PP::AbstractArray{<:GuidProp},
        XX, WW, ::Val{:ll}, y1::K, ::Val{true};
        Wnr=Wiener(), skip=0
    ) where K
    error("in-place not implemented")
end

#=---- with log-likelihood and Crank-Nicolson scheme ----=#
"""
    Random.rand!(
        [rng::Random.AbstractRNG]
        PP::AbstractArray{<:GuidProp},
        XX°, WW°, WW, ρρ, v::Val{:ll}, y1=zero(PP[1]);
        Wnr=Wiener()
    )

Sample a trajectory started from `y1` over multiple intervals for guided
proposals `PP` that correspond to consecutive intervals. Use containers `XX°`
and `WW°` to save the results. Use a preconditioned Crank-Nicolson scheme
with memory parameters `ρρ` (one for each interval) and a previously sampled
Wiener noise `WW`. Compute log-likelihood (path contribution AND end-points
contribution) along the way.
"""
function Random.rand!(
        PP::AbstractArray{<:GuidProp},
        XX°, WW°, WW, ρρ, v::Val{:ll}, y1=zero(PP[1]);
        Wnr=Wiener(), skip=0
    )
    rand!(
        Random.GLOBAL_RNG,
        PP, XX°, WW°, WW, ρρ, v, y1, DD.ismutable(y1);
        Wnr=Wnr, skip=skip
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        PP::AbstractArray{<:GuidProp},
        XX°, WW°, WW, ρρ, v::Val{:ll}, y1=zero(PP[1]);
        Wnr=Wiener(), skip=0
    )
    rand!(
        rng,
        PP, XX°, WW°, WW, ρρ, v, y1, DD.ismutable(y1);
        Wnr=Wnr, skip=skip
    )
end

function Random.rand!(
        rng::Random.AbstractRNG,
        PP::AbstractArray{<:GuidProp},
        XX°, WW°, WW, ρρ, v::Val{:ll}, y1::K, m::Val{false};
        Wnr=Wiener(), skip=0
    ) where K
    ll_tot = loglikhd_obs(PP[1], y1)
    for i in eachindex(PP)
        success, ll = rand!(
            rng, PP[i], XX°[i], WW°[i], WW[i], ρρ[i], v, y1, m;
            Wnr=Wnr, skip=skip
        )
        success || return false, ll
        ll_tot += ll
        y1 = XX°[i].x[end]
    end
    true, ll_tot
end


function Random.rand!(
        rng::Random.AbstractRNG,
        PP::AbstractArray{<:GuidProp},
        XX°, WW°, WW, ρρ, ::Val{:ll}, y1::K, ::Val{true};
        Wnr=Wiener(), skip=0
    ) where K
    error("in-place not implemented")
end


#===============================================================================
                                Forward guide
===============================================================================#
"""
alias to rand!
"""
const forward_guide! = rand!


#===============================================================================
                                Backward filter
===============================================================================#
"""
alias to `recompute_guiding_term!`
"""
const backward_filter! = recompute_guiding_term!
