#==============================================================================#
#
#                 some general & simple utility functions
#
#==============================================================================#

"""
    outer(x)

Compute an outer product
"""
outer(x) = x*x'


"""
    get_or_default(container, elem::Symbol, default)

Return `container`.`elem` if it exists, otherwise return `default`
"""
function get_or_default(container, elem::Symbol, default)
    haskey(container, elem) ? getindex(container, elem) : default
end


"""
    static_accessor_HFc(u::SVector, ::Val{T}) where T

Access data stored in the container `u` so that it matches the shapes of H,F,c
and points to the correct points in `u`. `T` is the dimension of the stochastic
process.
"""
function static_accessor_HFc(u::K, ::Val{T}) where {K<:Union{SVector,MVector},T}
    Hidx = SVector{T*T,Int64}(1:T*T)
    Fidx = SVector{T,Int64}((T*T+1):(T*T+T))
    #cidx = SVector{1,Int64}((T*T+T+1):(T*T+T+1))
    reshape(u[Hidx], Size(T,T)), u[Fidx], u[T*T+T+1]
end


@doc raw"""
    standard_guid_prop_time_transf(tt)

Standard time transformation used for guided proposals:
```math
τ: t → t_0 + (t-t_0)\frac{2 - (t-t_0)}{T-t_0},
```
applied to a vector `tt` where $t_0:=$`tt[1]` and $T:=$`tt[end]`
"""
standard_guid_prop_time_transf(tt) = τ(tt[1], tt[end]).(tt)
τ(t₀,T) = (t) ->  t₀ + (t-t₀) * (2-(t-t₀)/(T-t₀))
