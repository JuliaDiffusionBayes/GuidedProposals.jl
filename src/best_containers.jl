abstract type Container{T} <: AbstractArray{T,1} end

Base.size(var::Container) = size(var.data)
Base.eltype(var::Container{T}) where T = T

Base.getindex(var::Container, i::Int) = var.data[i]
Base.getindex(var::Container, I::Vararg{Int,N}) where N = var.data[I...]
Base.getindex(var::Container, ::Colon) = var.data[:]
Base.getindex(var::Container, kr::AbstractRange) = var.data[kr]

Base.setindex!(var::Container, v, i::Int) = (var.data[i] = v)
Base.setindex!(var::Container, v, I::Vararg{Int,N}) where N = (var.data[I...] = v)
Base.setindex!(var::Container, v, ::Colon) = (var.data[:] .= v)
Base.setindex!(var::Container, v, kr::AbstractRange) = (var.data[kr] .= v)


"""
    struct HFcContainer{T,D,TH,TF,Tc} <: AbstractBuffer{T}
        data::Vector{T}
        H::TH
        F::TF
        c::Tc
    end

A buffer containing data for in-place computations of H,F,c terms.
"""
struct HFcContainer{T,D,TH,TF,Tc} <: Container{T}
    data::Vector{T}
    H::TH
    F::TF
    c::Tc

    function HFcContainer(::Type{T}, D::Integer) where T
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
