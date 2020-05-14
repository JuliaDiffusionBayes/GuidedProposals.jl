
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
function ObservationSchemes.clone(P::GuidProp, θ°, η°)
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
function ObservationSchemes.clone(P::T, ξ, invcoords, θ°idx, η°idx) where T <: GuidProp
    T(
        DD.clone(P.P_target, ξ, invcoords, θ°idx, _BY_NAME),
        DD.clone(P.P_aux, ξ, invcoords, θ°idx, _BY_NAME),
        OBS.set_parameters!(P.obs, ξ, η°idx, _BY_POS),
        P.guiding_term_solver,
    )
end
