#=
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
function OBS.clone(P::T, θ°, η°) where T <: GuidProp
    #P_aux = DD.clone(P.P_aux, θ°)
    #P.guiding_term_solver.integrator.p = P_aux #NOTE let's force out-of-place for a moment
    out = T(
        DD.clone(P.P_target, θ°),
        DD.clone(P.P_aux, θ°),
        OBS.set_parameters!(P.obs, η°),
        P.guiding_term_solver,
    )
    out.guiding_term_solver.integrator.p = out.P_aux
    out
end

function clone!(PP::AbstractArray{<:GuidProp}, θ°, η°)
    for i in eachindex(PP)
        PP[i] = OBS.clone(PP[i], θ°, η°)
    end
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
    #P_aux = DD.clone(P.P_aux, ξ, invcoords, θ°idx, _BY_NAME)
    #P.guiding_term_solver.integrator.p = P_aux #NOTE let's force out-of-place for a moment
    out = T(
        DD.clone(P.P_target, ξ, invcoords, θ°idx, _BY_NAME),
        DD.clone(P.P_aux, ξ, invcoords, θ°idx, _BY_NAME),
        OBS.set_parameters!(P.obs, ξ, η°idx, _BY_POS),
        P.guiding_term_solver,
    )
    out.guiding_term_solver.integrator.p = out.P_aux
    out
end

function clone!(PP::AbstractArray{<:GuidProp}, ξ, invcoords, θ°idx, η°idxs)
    for i in eachindex(PP)
        PP[i] = OBS.clone(PP[i], ξ, invcoords, θ°idx, η°idxs[i])
    end
end


function OBS.set_parameters!(P::T, ξ, invcoords, θ°idx, η°idx) where T <: GuidProp
    #P_aux = DD.clone(P.P_aux, ξ, invcoords, θ°idx, _BY_NAME)
    #P.guiding_term_solver.integrator.p = P_aux #NOTE let's force out-of-place for a moment
    out = T(
        DD.clone(P.P_target, ξ, invcoords, θ°idx, _BY_NAME),
        DD.clone(P.P_aux, ξ, invcoords, θ°idx, _BY_NAME),
        OBS.set_parameters!(P.obs, ξ, η°idx, _BY_POS),
        P.guiding_term_solver,
    )
    out.guiding_term_solver.integrator.p = out.P_aux
    out
end

function clone!(PP::AbstractArray{<:GuidProp}, ξ, invcoords, θ°idx, η°idxs)
    for i in eachindex(PP)
        PP[i] = OBS.clone(PP[i], ξ, invcoords, θ°idx, η°idxs[i])
    end
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
    critical_parameters_changed(P::GuidProp, θ°idx, η°idx=nothing)

Return a boolean flag for whether the changed parameters (listed in `θ°idx` and
`η°idx`) prompt for re-computation of the guiding term.
"""
function critical_parameters_changed(P::T, θ°idx, η°idx=nothing) where T<:GuidProp
    critical_parameters_changed(T, θ°idx, η°idx)
end

function critical_parameters_changed(PP::AbstractArray{<:GuidProp}, θ°idx, η°idxs)
    any(
        x->( critical_parameters_changed(x[1], θ°idx, x[2]) ),
        zip(PP, η°idxs)
    )
end

"""
    critical_parameters_changed(_P_type::Type{<:GuidProp}, θ°idx, η°idx=nothing)

Return a boolean flag for whether the changed parameters (listed in `θ°idx` and
`η°idx`) prompt for re-computation of the guiding term.
"""
function critical_parameters_changed(_P_type::Type{<:GuidProp}, θ°idx, η°idx=nothing)
    η°idx === nothing || length(η°idx) == 0 || return true
    critical_p = critical_parameter_names(_P_type)
    any(i->(i.pname in critical_p), θ°idx)
end

function critical_parameters_changed(P::T, θ°::Dict, η°=nothing) where T <: GuidProp
    η° === nothing || length(η°) == 0 || return true
    critical_p = critical_parameter_names(P)
    any( k->( hasfield(T, k) && getfield(P, k) != θ°[k] ), keys(θ°) )
end

=#


"""
    DD.set_parameters!(
        PP::AbstractArray{<:GuidProp},
        PP°::AbstractArray{<:GuidProp},
        θ°,
        var_p_names,
        var_p_aux_names,
        θ_local_names,
        θ_local_aux_names,
        θ_local_obs_ind,
        critical_change = is_critical_update(
            PP, θ_local_aux_names, θ_local_obs_ind
        ),
    )

Set parameters of Guided proposals `PP°` in an MCMC setting.

...
# Arguments
- `PP`: containers indicating how `PP°` should look like before `θ°` update
- `θ°`: a vector with new parameters to be potentially set inside `PP°`
- `var_p_names`: list of all variable parameter names in `PP`
- `var_p_aux_names`: list of all variable parameter names of auxiliary laws in
`PP`
- `θ_local_names`: list of pairs with relevant `θ°` entries for the target law
in a format (idx-of-param-in-θ°::Int64, param-name-in-law::Symbol)
- `θ_local_aux_names`: list of pairs with relevant `θ°` entries for the aux law
in a format (idx-of-param-in-θ°::Int64, param-name-in-law::Symbol)
- `θ_local_obs_ind`: list of pairs with relevant `θ°` entries for the
observations in a format (idx-of-param-in-θ°::Int64, idx-in-obs::Int64)
- `critical_change`: boolean for whether `θ°` update alters any critical
parameters prompting for recomputation of the guiding term
...
"""
function DD.set_parameters!(
        PP::AbstractArray{<:GuidProp},
        PP°::AbstractArray{<:GuidProp},
        θ°,
        var_p_names,
        var_p_aux_names,
        θ_local_names,
        θ_local_aux_names,
        θ_local_obs_ind,
        critical_change = is_critical_update(
            PP, θ_local_aux_names, θ_local_obs_ind
        ),
    )
    # Resetting the state from the previous update
    # --------------------------------------------
    # first, equalize observations
    equalize_obs_params!(PP, PP°) && (critical_change = true)

    # second, equalize parameters
    if !DD.same_entries(PP, PP°, var_p_names)
        equalize_law_params!(
            PP, PP°, var_p_names, var_p_aux_names
        ) && (critical_change = true)
    end

    # Setting the new parameters
    # --------------------------
    DD.set_parameters!(
        PP°, θ°, θ_local_names, θ_local_aux_names, θ_local_obs_ind
    )
    critical_change
end

"""
    equalize_obs_params!(
        PP::AbstractArray{<:GuidProp},
        PP°::AbstractArray{<:GuidProp}
    )

Go through a collection of `GuidProp` in PP° and make sure that all observation
parameters are the same. If not, equalize them and return a `critical_change`
flag indicating that `GuidProp` laws need to be recomputed.
"""
function equalize_obs_params!(
        PP::AbstractArray{<:GuidProp},
        PP°::AbstractArray{<:GuidProp},
    )
    critical_change = false
    for i in eachindex(PP, PP°)
        if PP[i].obs.θ != PP°[i].obs.θ
            critical_change = true
            PP°[i].obs.θ .= PP[i].obs.θ
        end
    end
    critical_change
end

"""
    DD.same_entries(
        PP::AbstractArray{<:GuidProp},
        PP°::AbstractArray{<:GuidProp},
        entries
    )
Go through the collections of guided proposals `PP` and `PP°` and compare if
they share the same values of fields listed in `entries`.
"""
function DD.same_entries(
        PP::AbstractArray{<:GuidProp},
        PP°::AbstractArray{<:GuidProp},
        entries
    )
    DD.same_entries(PP[1].P_target, PP°[1].P_target, entries)
end

"""
    equalize_law_params!(
        PP::AbstractArray{<:GuidProp},
        PP°::AbstractArray{<:GuidProp},
        var_p_names,
        var_p_aux_names,
    )

Go through two collections of `GuidProp` and compare their variable parameters.
If any of the parameters in `PP°` ends up being different than in `PP` then
equalize them and if any of them belongs to the auxiliary law then return a
`critical_change` flag.
"""
function equalize_law_params!(
        PP::AbstractArray{<:GuidProp},
        PP°::AbstractArray{<:GuidProp},
        var_p_names,
        var_p_aux_names,
    )
    critical_change = false
    for i in eachindex(PP, PP°)
        DD.set_parameters!(PP[i].P_target, PP°[i].P_target, var_p_names)
        ( # check for a critical change
            critical_change ||
            DD.same_entries(PP[i].P_aux, PP°[i].P_aux, var_p_aux_names[i]) ||
            (critical_change = true)
        )
        DD.set_parameters!(PP[i].P_aux, PP°[i].P_aux, var_p_aux_names[i])
    end
    return critical_change
end

"""
    DD.set_parameters!(
        PP::AbstractArray{<:GuidProp},
        θ°,
        θ_local_names,
        θ_local_aux_names,
        θ_local_obs_ind
    )

Go throught the collection of `GuidProp`s in `PP` and set the new parameters
`θ°` in relevant fields. `θ_local_names` lists which fields in the target
law need changing. `θ_local_aux_names` lists the same for the auxiliary laws
and `θ_local_obs_ind` for the observations.
"""
function DD.set_parameters!(
        PP::AbstractArray{<:GuidProp},
        θ°,
        θ_local_names,
        θ_local_aux_names,
        θ_local_obs_ind
    )
    for i in eachindex(PP)
        DD.set_parameters!(
            PP[i], θ°, θ_local_names, θ_local_aux_names[i], θ_local_obs_ind[i]
        )
    end
end

"""
    DD.set_parameters!(
        P::GuidProp,
        θ°,
        θ_local_names,
        θ_local_aux_names,
        θ_local_obs_ind
    )

Set the new parameters `θ°` in `P`. `θ_local_names` lists which fields in the
target law need changing. `θ_local_aux_names` lists the same for the auxiliary
law and `θ_local_obs_ind` for the observation.
"""
function DD.set_parameters!(
        P::GuidProp,
        θ°,
        θ_local_names,
        θ_local_aux_names,
        θ_local_obs_ind
    )
    DD.set_parameters!(P.P_target, θ°, θ_local_names)
    DD.set_parameters!(P.P_aux, θ°, θ_local_aux_names)
    OBS.set_parameters!(P.obs, θ°, θ_local_obs_ind)
    P.guiding_term_solver.integrator.p = P.P_aux # for safe measure
end

"""
    is_critical_update(
        PP::AbstractArray{<:GuidProp},
        θ_local_aux_names,
        θ_local_obs_ind
    )

Check if the update of parameters that updates the fields of the auxiliary law
listed in `θ_local_aux_names` and the observations listed in `θ_local_obs_ind`
is critical i.e. whether it prompts for recomputation of the guiding term.
"""
function is_critical_update(
        PP::AbstractArray{<:GuidProp},
        θ_local_aux_names,
        θ_local_obs_ind
    )
    for i in eachindex(PP)
        is_critical_update(
            PP[i], θ_local_aux_names[i], θ_local_obs_ind[i]
        ) && return true
    end
    false
end

function is_critical_update(P::GuidProp, θ_local_aux_names, θ_local_obs_ind)
    DD.has_any(P.P_aux, θ_local_aux_names) || length(θ_local_obs_ind) > 0
end

"""
    DD.set_parameters!(P::GuidProp, θ°::Dict)

Convenience parameter setter
"""
function DD.set_parameters!(P::GuidProp, θ°::Dict)
    DD.set_parameters!(P.P_target, θ°)
    DD.set_parameters!(P.P_aux, θ°)
    P.guiding_term_solver.integrator.p = P.P_aux # for safe measure
end

"""
    DD.set_parameters!(PP::AbstractArray{<:GuidProp}, θ°::Dict)

Convenience parameter setter
"""
function DD.set_parameters!(PP::AbstractArray{<:GuidProp}, θ°::Dict)
    for i in eachindex(PP)
        DD.set_parameters!(PP[i], θ°)
    end
end

set_aux_obs!(P::GuidProp, obs) = (P.aux_obs.obs .= obs)
