"""
    struct HFcContainer{T,D,TH,TF,Tc} <: AbstractBuffer{T}
        data::Vector{T}
        H::TH
        F::TF
        c::Tc
    end

A buffer containing data for in-place computations of H,F,c terms.
"""
struct HFcContainer{T,D,TH,TF,Tc} <: DiffusionDefinition.AbstractBuffer{T}
    data::Vector{T}
    H::TH
    F::TF
    c::Tc

    function HFcContainer{T}(D::Integer) where T
        data = zeros(T, D*D+D+1)
        H = reshape( view(data, 1:(D*D)), (D,D) )
        F = view(data, (D*D+1):(D*D+D))
        c = view(data, (D*D+D+1):(D*D+D+1))
        new{T,D,typeof(H),typeof(F),typeof(c)}(data, H, F, c)
    end

    function HFcContainer(data::AbstractArray{T,1}, D) where T
        H = reshape( view(data, 1:(D*D)), (D,D) )
        F = view(data, (D*D+1):(D*D+D))
        c = view(data, (D*D+D+1):(D*D+D+1))
        new{T,D,typeof(H),typeof(F),typeof(c)}(data, H, F, c)
    end
end

function Base.similar(c::HFcContainer{T,D}) where {T,D}
    HFcContainer(similar(c.data), D)
end

function Base.similar(c::HFcContainer{T,D}, ::Type{K}) where {T,D,K}
    HFcContainer(similar(c.data, K), D)
end

"""
    struct HFcBuffer{
            T,D,TB,Tβ,Tσ,Ta,Tmat,Tvec
            } <: DiffusionDefinition.AbstractBuffer{T}
        data::Vector{T}
        B::TB
        β::Tβ
        σ::Tσ
        a::Ta
        mat::Tmat
        vec::Tvec
    end

A buffer for temporary computations of in-place ODE solvers solving for H,F,c
system.
"""
struct HFcBuffer{
        T,D,TB,Tβ,Tσ,Ta,Tmat,Tvec
        } <: DiffusionDefinition.AbstractBuffer{T}
    data::Vector{T}
    B::TB
    β::Tβ
    σ::Tσ
    a::Ta
    mat::Tmat
    vec::Tvec

    function HFcBuffer{T}(D::Integer) where T
        data = zeros(T, 4*D*D+D)

        B = reshape( view(data, 1:(D*D)), (D,D) )
        β = view(data, (D*D+1):(D*D+D))
        σ = reshape( view(data, (D*D+D+1):(2*D*D+D)), (D,D) )
        a = reshape( view(data, (2*D*D+D+1):(3*D*D+D)), (D,D) )
        m = reshape( view(data, (3*D*D+D+1):(4*D*D+D)), (D,D) )
        v =  view(data, (3*D*D+D+1):(3*D*D+2*D))
        new{T,D,typeof(B),typeof(β),typeof(σ),typeof(a),typeof(m),typeof(v)}(
            data, B, β, σ, a, m, v
        )
    end
end
