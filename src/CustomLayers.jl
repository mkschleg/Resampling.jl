
mutable struct SingleLayer{F, FP, A}
    σ::F
    σ′::FP
    W::A
end

SingleLayer(in::Integer, out::Integer, σ, σ′; init=(dims...)->zeros(Float32, dims...)) =
    SingleLayer(σ, σ′, init(out, in))

(layer::SingleLayer)(x) = layer.σ.(layer.W*x)
deriv(layer::SingleLayer, x) = layer.σ′.(layer.W*x)

# Sparse Updates

Linear(in::Integer, out::Integer) =
    SingleLayer(in, out, identity, (x)->1.0)

# SparseLayer(num_weights, args...; kwargs...) =
#     SingleLayer(num_weights, out, args...; kwargs...)

# sigmoid = Flux.sigmoid
sigmoid′(x) = sigmoid(x)*(1.0-sigmoid(x))
identity′(x) = 1.0
tanh′(x) = 1 - (tanh(x))^2

mutable struct TabularLayer{A}
    W::A
end
Tabular(dims::Integer...) = TabularLayer(zeros(dims...))

@forward TabularLayer.W Base.getindex, Base.setindex!


(layer::TabularLayer)(x) = layer[x]
(layer::TabularLayer)(x::Array{T, 1}) where {T<:Integer} = layer[x...]

mutable struct SparseLayer{F, FP, A, P}
    σ::F
    σ′::FP
    W::A
    ϕ::P
end

SparseLayer(num_weights::Integer, out::Integer, σ=identity, σ′=identity′; init=(dims...)->zeros(Float32, dims...)) =
    SparseLayer(σ, σ′, init(out, num_weights), zeros(Float32, out, num_weights))

(layer::SparseLayer)(x::Array{Int64, 1}) = layer.σ.(sum(layer.W[x]))
deriv(layer::SparseLayer, x::Array{Int64, 1}) = layer.σ′.(sum(layer.W[x]))

(layer::SparseLayer)(x::Array{CartesianIndex{1}, 1}) = layer.σ.(sum(layer.W[x]))
deriv(layer::SparseLayer, x::Array{CartesianIndex{1}, 1}) = layer.σ′.(sum(layer.W[x]))

