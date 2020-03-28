#==============================================================================#
#
#    Generated functions for convenient access of data by ODE solvers for:
#    - M, L, μ
#    - H, F, c
#    - P, ν
#    It assumes that all elements are stored in a one-dimensional array
#    and functions _M, _L, etc. are provided that access specific data
#    segments and reshape them appropriately
#
#==============================================================================#


#------------------------------------------------------------------------------#
#                            For M, L, μ solvers
#
#       Assumes that data is stored in data in a format:
# data: |  <---m^2--->  |  <---m*d--->  | <---m--->  |
#       |---------------|---------------|------------|
#       |   matrix M    |    matrix L   |  vector μ  |
#
#------------------------------------------------------------------------------#

const OUT_VEC_TYPE{T} = SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true} where T
const OUT_MAT_TYPE{T} = Base.ReshapedArray{T,2,OUT_VEC_TYPE{T},Tuple{}}

"""
    _M(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a matrix `M` that are stored in a container `data`
"""
function _M(data::Vector{K}, ::Val{T})::OUT_MAT_TYPE{K} where {K,T}
    reshape(
        view(
            data,
            1:(T[2]*T[2])
        ),
        (T[2],T[2])
    )
end

"""
    _L(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a matrix `L` that are stored in a container `data`
"""
function _L(data::Vector{K}, ::Val{T})::OUT_MAT_TYPE{K} where {K,T}
    reshape(
        view(
            data,
            (T[2]*T[2]+1):((T[2]+T[1])*T[2])
        ),
        (T[2],T[1])
    )
end

"""
    _μ(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a vector `μ` that are stored in a container `data`
"""
function _μ(data::Vector{K}, ::Val{T})::OUT_VEC_TYPE{K} where {K,T}
    view(
        data,
        ((T[2]+T[1])*T[2]+1):((T[2]+T[1]+1)*T[2])
    )
end

#------------------------------------------------------------------------------#
#                            For H, F, c solvers
#
#       Assumes that data is stored in data in a format:
# data: |  <---d^2--->  |  <---d--->  | <---1--->  |  <--- d^2 ---> |
#       |               |             |            | <---d--->|
#       |---------------|-------------|------------|          |     |
#       |   matrix H    |   vector F  |  scalar c  | temp matrix    |
#       |               |             |            | temp vec |
#
#------------------------------------------------------------------------------#
#=
"""
    _H(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a matrix `H` that are stored in a container `data`
"""
function _H(data::Vector{K}, ::Val{T})::OUT_MAT_TYPE{K} where {K,T}
    reshape(
        view(
            data,
            1:(T*T)
        ),
        (T,T)
    )
end

"""
    _F(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a vector `F` that are stored in a container `data`
"""
function _F(data::Vector{K}, ::Val{T})::OUT_VEC_TYPE{K} where {K,T}
    view(
        data,
        (T*T+1):(T*T+T)
    )
end

"""
    _c(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a scalar `c` that is stored in a container `data`
"""
function _c(data::Vector{K}, ::Val{T})::OUT_VEC_TYPE{K} where {K,T}
    view(
        data,
        (T*T+T+1):(T*T+T+1)
    )
end
=#

#------------------------------------------------------------------------------#
#                              For P, ν solvers
#------------------------------------------------------------------------------#

"""
    _P(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a matrix `P` that are stored in a container `data`
"""
function _P(data::Vector{K}, ::Val{T})::OUT_MAT_TYPE{K} where {K,T}
    reshape(
        view(
            data,
            1:(T*T)
        ),
        (T,T)
    )
end

"""
    _ν(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a vector `ν` that are stored in a container `data`
"""
function _ν(data::Vector{K}, ::Val{T})::OUT_VEC_TYPE{K} where {K,T}
    view(
        data,
        (T*T+1):(T*(T+1))
    )
end


#------------------------------------------------------------------------------#
#                        For accessing B, β, σ and a
#------------------------------------------------------------------------------#
#=
"""
    _B(buffer::AbstractArray, ::Val{T}) where T

Provide a view to contents of a matrix `B` that are stored in a container
`buffer`
"""
function _B(buffer::Vector{K}, ::Val{T})::OUT_MAT_TYPE{K} where {K,T}
    reshape(
        view(
            buffer,
            1:(T*T)
        ),
        (T,T)
    )
end

"""
    _β(buffer::AbstractArray, ::Val{T}) where T

Provide a view to contents of a vector `β` that are stored in a container
`buffer`
"""
function _β(buffer::Vector{K}, ::Val{T})::OUT_VEC_TYPE{K} where {K,T}
    view(
        buffer,
        (T*T+1):(T*T+T)
    )
end

"""
    _σ(buffer::AbstractArray, ::Val{T}) where T

Provide a view to contents of a matrix `σ` that are stored in a container
`buffer`
"""
function _σ(buffer::Vector{K}, ::Val{T})::OUT_MAT_TYPE{K} where {K,T}
    reshape(
        view(
            buffer,
            (T*T+T+1):(2*T*T+T)
        ),
        (T,T)
    )
end

"""
    _a(buffer::AbstractArray, ::Val{T}) where T

Provide a view to contents of a matrix `a` that are stored in a container
`buffer`
"""
function _a(buffer::Vector{K}, ::Val{T})::OUT_MAT_TYPE{K} where {K,T}
    reshape(
        view(
            buffer,
            (2*T*T+T+1):(3*T*T+T)
        ),
        (T,T)
    )
end
=#
#------------------------------------------------------------------------------#
#                  Accessing additional, temporary storage
#------------------------------------------------------------------------------#
#=
"""
    _temp_matH(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a temporary matrix, stored in a container
`buffer`. NOTE: `_temp_matH` and `_temp_vecH` access overlapping chunks of
memory.
"""
function _temp_matH(buffer::Vector{K}, ::Val{T})::OUT_MAT_TYPE{K} where {K,T}
    reshape(
        view(
            buffer,
            (3*T*T+T+1):(4*T*T+T)
        ),
        (T,T)
    )
end

"""
    _temp_vecH(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a temporary vector, stored in a container
`buffer`. NOTE: `_temp_matH` and `_temp_vecH` access overlapping chunks of
memory.
"""
function _temp_vecH(buffer::Vector{K}, ::Val{T})::OUT_VEC_TYPE{K} where {K,T}
    view(
        buffer,
        (3*T*T+T+1):(3*T*T+2*T)
    )
end
=#
#==============================================================================#
#
#                            STATIC ACCESSORS
#
#==============================================================================#


#------------------------------------------------------------------------------#
#                            For H, F, c solvers
#------------------------------------------------------------------------------#
function static_accessor_HFc(u::AbstractArray, ::Val{T}) where T
    Hidx = SVector{T*T,Int64}(1:T*T)
    Fidx = SVector{T,Int64}((T*T+1):(T*T+T))
    reshape(u[Hidx], Size(T,T)), u[Fidx], u[T*T+T+1]
end


#==============================================================================#
#
#                      ACCESSORS FOR PATH SIMULATIONS
#
#==============================================================================#

#------------------------------------------------------------------------------#
#                                Default
#------------------------------------------------------------------------------#
function _dW_sim(buffer::Vector{K}, ::Val{T})::OUT_VEC_TYPE{K} where {K,T}
    view(
        buffer,
        1:T[2]
    )
end

function _σ_sim(buffer::Vector{K}, ::Val{T})::OUT_MAT_TYPE{K} where {K,T}
    reshape(
        view(
            buffer,
            (T[2]+1):(T[2]+T[1]*T[2])
        ),
        T
    )
end

function _b_sim(buffer::Vector{K}, ::Val{T})::OUT_VEC_TYPE{K} where {K,T}
    view(
        buffer,
        (T[2]+T[1]*T[2]+1):(T[2]+T[1]*T[2]+T[1])
    )
end

#------------------------------------------------------------------------------#
#                           Guided Proposals
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#                           Linear Diffusions
#------------------------------------------------------------------------------#
function _B_sim(buffer::Vector{K}, v::Val{T})::OUT_MAT_TYPE{K} where {K,T}
    reshape(
        view(
            buffer,
            (T[2]+T[1]*T[2]+T[1]+1):(T[2]+T[1]*T[2]+T[1]+T[1]*T[1])
        ),
        (T[1],T[1])
    )
end

function _β_sim(buffer::Vector{K}, v::Val{T})::OUT_VEC_TYPE{K} where {K,T}
    view(
        buffer,
        (T[2]+T[1]*T[2]+T[1]+T[1]*T[1]+1):(T[2]+T[1]*T[2]+T[1]+T[1]*T[1]+T[1])
    )
end
