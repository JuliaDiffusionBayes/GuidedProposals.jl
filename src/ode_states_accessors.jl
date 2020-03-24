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
#------------------------------------------------------------------------------#


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
#------------------------------------------------------------------------------#

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

@generated function _F(data::AbstractArray, ::Val{T}) where T
    Expr(
        :call,
        :view,
        :data,
        (T*T+1):(T*T+T)
    )
end

@generated function _c(data::AbstractArray, ::Val{T}) where T
    Expr(
        :call,
        :view,
        :data,
        (T*T+T+1):(T*T+T+1)
    )
end


#------------------------------------------------------------------------------#
#                              For P, ν solvers
#------------------------------------------------------------------------------#

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

@generated function _ν(data::AbstractArray, ::Val{T}) where T
    Expr(
        :call,
        :view,
        :data,
        (T*T+1):(T*(T+1))
    )
end
