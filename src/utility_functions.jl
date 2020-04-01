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
