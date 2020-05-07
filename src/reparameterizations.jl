
"""
    GuidProp(P::GuidProp, θ°, η°)

Create new guided proposals object with new parameters `θ°` parametrizing the
diffusion laws and `η°` parametrizing the observations. Keep the pre-allocated
spaces for solvers unchanged.
"""
function GuidProp(P::GuidProp, θ°, η°)
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

function OBS.clone(P::GuidProp, ξ, invcoords, θ°idx, η°idx)
    T = typeof(P)
    T(
        DD.clone(P.P_target, ξ, invcoords, θ°idx, _BY_NAME),
        DD.clone(P.P_aux, ξ, invcoords, θ°idx, _BY_NAME),
        OBS.set_parameters!(P.obs, ξ, η°idx, _BY_POS),
        P.guiding_term_solver,
    )
end
