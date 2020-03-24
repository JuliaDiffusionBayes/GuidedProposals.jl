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

"""
    _M(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a matrix `M` that are stored in a container `data`
"""
@generated function _M(data::AbstractArray, ::Val{T}) where T
    Expr(
        :call,
        :reshape,
        Expr(
            :call,
            :view,
            :data,
            1:(T[2]*T[2])
        ),
        (T[2],T[2])
    )
end

"""
    _L(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a matrix `L` that are stored in a container `data`
"""
@generated function _L(data::AbstractArray, ::Val{T}) where T
    Expr(
        :call,
        :reshape,
        Expr(
            :call,
            :view,
            :data,
            (T[2]*T[2]+1):((T[2]+T[1])*T[2])
        ),
        (T[2],T[1])
    )
end

"""
    _μ(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a vector `μ` that are stored in a container `data`
"""
@generated function _μ(data::AbstractArray, ::Val{T}) where T
    Expr(
        :call,
        :view,
        :data,
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

"""
    _H(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a matrix `H` that are stored in a container `data`
"""
@generated function _H(data::AbstractArray, ::Val{T}) where T
    Expr(
        :call,
        :reshape,
        Expr(
            :call,
            :view,
            :data,
            1:(T*T)
        ),
        (T,T)
    )
end

"""
    _F(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a vector `F` that are stored in a container `data`
"""
@generated function _F(data::AbstractArray, ::Val{T}) where T
    Expr(
        :call,
        :view,
        :data,
        (T*T+1):(T*T+T)
    )
end

"""
    _c(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a scalar `c` that is stored in a container `data`
"""
@generated function _c(data::AbstractArray, ::Val{T}) where T
    Expr(
        :call,
        :view,
        :data,
        (T*T+T+1):(T*T+T+1)
    )
end

"""
    _temp_matH(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a temporary matrix, stored in a container `data`.
"""
@generated function _temp_matH(data::AbstractArray, ::Val{T}) where T
    Expr(
        :call,
        :reshape,
        Expr(
            :call,
            :view,
            :data,
            (T*T+T+2):(2*T*T+T+1)
        ),
        (T,T)
    )
end

"""
    _temp_vecH(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a temporary vector, stored in a container `data`.
"""
@generated function _temp_vecH(data::AbstractArray, ::Val{T}) where T
    Expr(
        :call,
        :view,
        :data,
        (T*T+T+2):(T*T+2*T+1)
    )
end


#------------------------------------------------------------------------------#
#                              For P, ν solvers
#------------------------------------------------------------------------------#

"""
    _P(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a matrix `P` that are stored in a container `data`
"""
@generated function _P(data::AbstractArray, ::Val{T}) where T
    Expr(
        :call,
        :reshape,
        Expr(
            :call,
            :view,
            :data,
            1:(T*T)
        ),
        (T,T)
    )
end

"""
    _ν(data::AbstractArray, ::Val{T}) where T

Provide a view to contents of a vector `ν` that are stored in a container `data`
"""
@generated function _ν(data::AbstractArray, ::Val{T}) where T
    Expr(
        :call,
        :view,
        :data,
        (T*T+1):(T*(T+1))
    )
end
