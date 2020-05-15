
"""
    ObservationSchemes.clone(P::GuidProp, θ°, η°)

Create a new guided proposal object with new parameters `θ°` parametrizing the
diffusion laws and `η°` parametrizing the observations. Keep the pre-allocated
spaces for solvers unchanged. Note that `θ°` must be a dictionary correspond to
parameters returned when calling DD.var_parameters() on the target and auxiliary
laws.

TODO this is a convenience method that is not really used and doesn't properly
work yet.
"""
function OBS.clone(P::GuidProp, θ°, η°)
    T = typeof(P)
    T(
        DD.clone(P.P_target, θ°),
        DD.clone(P.P_aux, θ°),
        OBS.clone(P.obs, η°),
        P.guiding_term_solver,
    )
end

const _BY_NAME = Val(:associate_by_name)
const _BY_POS = Val(:associate_by_position)

"""
    ObservationSchemes.clone(P::GuidProp, ξ, invcoords, θ°idx, η°idx)

Create a new guided proposal object with new parameters (slightly akward method,
but optimized for speed in an MCMC setting). `ξ` should contain all parameters
that are changing. It is a sub-vector of a larger `global` vector `ξ°`. The
positions taken by `ξ` in `ξ°` are given by `sub_idx` (which is not known to a
method), and `invcoords` is a dictionary that for each entry in `sub_idx` gives
a corresponding index of `ξ`. The indices of `ξ°` that constitute parameters to
be passed to cloning of law are listed in `θ°idx`. In particular, if `θ°idx`
were just a list of indices, then `θ° = ξ°[θ°idx]`. However, `θ°idx`
additionally contains parameter names. `η°idx` does the same but for the
observation. Keeps the pre-allocated spaces for solvers unchanged.
"""
function OBS.clone(P::T, ξ, invcoords, θ°idx, η°idx) where T <: GuidProp
    T(
        DD.clone(P.P_target, ξ, invcoords, θ°idx, _BY_NAME),
        DD.clone(P.P_aux, ξ, invcoords, θ°idx, _BY_NAME),
        OBS.set_parameters!(P.obs, ξ, η°idx, _BY_POS),
        P.guiding_term_solver,
    )
end

"""
    critical_parameter_names(P::GuidProp)

Return a list of parameter names that—if changed—prompt for re-computation
of a guiding term.
"""
critical_parameter_names(P::GuidProp) = critical_parameter_names(typeof(P))


"""
    critical_parameter_names(::Type{T}) where T<:GuidProp

Return a list of parameter names that—if changed—prompt for re-computation
of a guiding term.
"""
function critical_parameter_names(
        ::Type{P}
    ) where {P<:GuidProp{K,DP,DW,SS,R,R2,O,S,T}} where {K,DP,DW,SS,R,R2,O,S,T}
    DD.var_parameter_names(R2)
end

"""
    critical_parameters_changed(P::GuidProp, θ°idx, η°idx)

Return a boolean flag for whether the changed parameters (listed in `θ°idx` and
`η°idx`) prompt for re-computation of the guiding term.
"""
function critical_parameters_changed(P::GuidProp, θ°idx, η°idx)
    critical_parameters_changed(typeof(P), θ°idx, η°idx)
end

"""
    critical_parameters_changed(_P_type::Type{<:GuidProp}, θ°idx, η°idx)

Return a boolean flag for whether the changed parameters (listed in `θ°idx` and
`η°idx`) prompt for re-computation of the guiding term.
"""
function critical_parameters_changed(_P_type::Type{<:GuidProp}, θ°idx, η°idx)
    length(η°idx) == 0 || return true
    critical_p = critical_parameter_names(_P_type)

    for i in θ°idx
        i.pname in critical_p && return true
    end
    false
end
