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
        TB,Tβ,Tσ,Ta,Tmat,Tvec
        } <: DiffusionDefinition.AbstractBuffer
    B::TB
    β::Tβ
    σ::Tσ
    a::Ta
    mat::Tmat
    vec::Tvec

    function HFcBuffer(
            B::TB, β::Tβ, σ::Tσ, a::Ta, mat::Tmat, vec::Tvec
        ) where {TB,Tβ,Tσ,Ta,Tmat,Tvec}
        new{TB,Tβ,Tσ,Ta,Tmat,Tvec}(B, β, σ, a, mat, vec)
    end
end

function HFcBuffer{T}(D::Integer) where T
    B = zeros(T, (D, D))
    β = zeros(T, D)
    σ = zeros(T, (D, D))
    a = zeros(T, (D, D))
    m = zeros(T, (D, D))
    v = zeros(T, D)

    HFcBuffer(B, β, σ, a, m, v)
end


#=

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

=#

#=
"""
    struct HFcContainer{T,D,TH,TF,Tc} <: AbstractBuffer{T}
        data::Vector{T}
        H::TH
        F::TF
        c::Tc
    end

A buffer containing data for in-place computations of H,F,c terms.
"""
struct HFcContainer{T,TD,TH,TF,Tc} <: DiffusionDefinition.AbstractBuffer{T}
    data::TD
    H::TH
    F::TF
    c::Tc

    function HFcContainer{T}(
            data::TD, H::TH, F::TF, c::Tc
        ) where {T,TD,TH,TF,Tc}
        new{T,TD,TH,TF,Tc}(data, H, F, c)
    end
end

function HFcContainer{T}(D::Integer) where T
    H = zeros(T, (D,D))
    F = zeros(T, D)
    c = zeros(T, 1)
    data = ArrayPartition(H, F, c)
    HFcContainer{T}(data, H, F, c)
end

function HFcContainer(data::ArrayPartition{T}) where T
    HFcContainer{T}(data, data.x[1], data.x[2], data.x[3])
end

function Base.similar(d::HFcContainer{T}) where T
    new_data = similar(d.data)
    HFcContainer{T}(new_data, new_data.x[1], new_data.x[2], new_data.x[3])
end
# the one below is likely wrong, but it's also not used atm, come back later
function Base.similar(d::HFcContainer{K}, ::Type{T}) where {K,T}
    new_data = similar(d.data, T)
    HFcContainer{T}(new_data, new_data.x[1], new_data.x[2], new_data.x[3])
end


#function Base.similar(c::HFcContainer{T,D}) where {T,D}
#    HFcContainer(similar(c.data), D)
#end

#function Base.similar(c::HFcContainer{T,D}, ::Type{K}) where {T,D,K}
#    HFcContainer(similar(c.data, K), D)
#end
=#
