#===============================================================================

    Implementation of `backward filter` and `forward guide`---the two
    main workhorse routines of Guided Proposals.

===============================================================================#
struct IndexableNothing end
Base.getindex(i_n::IndexableNothing, i) = i_n
const Mutable = Array

#===============================================================================
                                Forward guide
===============================================================================#

function forward_guide!(
        WW°::Vector{<:Trajectory},
        XX::Vector{<:Trajectory},
        Wnr::Vector{<:Wiener},
        PP::Vector{<:GuidProp},
        x0,
        WW=IndexableNothing(),
        ρs=IndexableNothing(),
    )
    num_segments = length(WW°)
    ll°_tot = 0.0
    for i in 1:num_segments
        success, ll° = forward_guide!(
            WW°[i], XX[i], Wnr[i], PP[i], x0, WW[i], ρs[i]
        )
        success || return false, -Inf
        x0 = XX[i].x[end]
        ll°_tot += ll°
    end
    true, ll°_tot
end

function forward_guide!(
        W°::Trajectory, X::Trajectory, Wnr::Wiener, P::GuidProp, x0
    )
    forward_guide!(W°, X, Wnr, P, x0, IndexableNothing(), IndexableNothing())
end

function forward_guide!(
        W°::Trajectory, X::Trajectory, Wnr::Wiener, P::GuidProp, x0,
        ::IndexableNothing, ::IndexableNothing,
    )
    rand!(W°, Wnr)
    solve_and_ll!(X, W°, P, x0)
end


function forward_guide!(
        W°::Trajectory, X::Trajectory, Wnr::Wiener, P::GuidProp, x0, W, ρ,
    )
    rand!(W°, Wnr)
    crank_nicolson!(W°.x, W.x, ρ)
    solve_and_ll!(X, W°, P, x0)
end

function crank_nicolson!(y°::Vector, y, ρ) # For immutable types
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
                                Backward filter
===============================================================================#

function backward_filter!(P::Vector{<:GuidProp})
    N = length(P)
    recompute_guiding_term!(P[end])
    for i in (N-1):-1:1
        recompute_guiding_term!(P[i], P[i+1])
    end
end
