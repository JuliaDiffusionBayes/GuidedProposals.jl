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

function set_obs!(P::GuidProp, obs)
    P.obs.obs .= obs
    P.P_aux.vT .= obs
    P.guiding_term_solver.integrator.p = P.P_aux
end
