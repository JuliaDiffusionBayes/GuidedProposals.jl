function HFcContainer(::Type{K}, D) where K
    ArrayPartition((zeros(K,(D,D)),zeros(K,D), zeros(K,1)))
end

#=

struct HFcContainer{D,TH,TF,Tc}
    H::TH
    F::TF
    c::Tc

    function HFcContainer(H::TH, F::TF, c::Tc) where {TH,TF,Tc}
        D = length(F)
        new{D,TH,TF,Tc}(H, F, c)
    end
end

function HFcContainer(::Type{T}, D::Integer) where T
    H = zeros(T, (D,D))
    F = zeros(T, D)
    c = zeros(T, 1)
    HFcContainer(H, F, c)
end

Base.size(var::HFcContainer{D}) where D = ((D,D),D,1)
Base.eltype(var::HFcContainer) = eltype(var.H)
Base.length(var::HFcContainer{D}) where D = D*(D+1)+1

function Base.getindex(var::HFcContainer{D}, i::Int) where D
    i <= D*D ? var.H[i] : (i <= D*(D+1) ? var.F[i-D*D] : var.c[i-D*(D+1)])
end

function Base.getindex(var::HFcContainer, i::Int, I::Vararg{Int,N}) where N
    i == 1 ? var.H[I...] : ( i== 2 ? var.F[I...] : var.c[I...] )
end

function Base.getindex(var::HFcContainer, ::Colon)
    var.H[:], var.F[:], var.c[:]
end

function Base.setindex!(var::HFcContainer{D}, v, i::Int) where D
    (
        i <= D*D ?
        (var.H[i]=v) :
        (
            i <= D*(D+1) ?
            (var.F[i-D*D]=v) :
            (var.c[i-D*(D+1)]=v)
        )
    )
end

function Base.setindex!(var::HFcContainer, v, i::Int, I::Vararg{Int,N}) where N
    (
        i == 1 ?
        (var.H[I...] = v) :
        (
            i == 2 ?
            (var.F[I...] = v) :
            (var.c[I...] = v)
        )
    )
end

function Base.setindex!(var::HFcContainer, v, ::Colon)
    var.H[:], var.F[:], var.c[:] .= v
end

function Base.similar(var::HFcContainer{D}) where D
    HFcContainer(similar(var.H), similar(var.F), similar(var.c))
end

function Base.similar(var::HFcContainer{D}, ::Type{K}) where {D,K}
    HFcContainer(similar(var.H, K), similar(var.F, K), similar(var.c, K))
end

function Base.zero(var::HFcContainer{D}) where D
    HFcContainer(zero(var.H), zero(var.F), zero(var.c))
end

OrdinaryDiffEq.recursive_bottom_eltype(a::HFcContainer) = eltype(a.H)
OrdinaryDiffEq.recursive_unitless_bottom_eltype(a::HFcContainer) = eltype(a.H)
=#
