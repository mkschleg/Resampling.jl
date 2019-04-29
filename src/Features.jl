module Features
import Combinatorics: combinations

abstract type AbstractFeatures end

struct RandomFeatures{T, F<:AbstractFloat} <: AbstractFeatures
    # feature_map::Array{Array{Float64, 1}, 1}
    feature_map::Dict{T, Array{Float64, 1}}
    function RandomFeatures{F}(state_list::Array{T, 1}, feature_size, active_features; rng=Random.GLOBAL_RNG) where {T, F<:AbstractFloat}
        comb = shuffle(rng, collect(combinations(1:feature_size, active_features)))
        features = [zeros(F, feature_size) for i in 1:chain_size]
        for (idx, state) in enumerate(state_list)
            features[state[comb[idx]] .= 1.0
        end
        new(features)
    end
end

@forward RandomFeatures.features Base.getindex

struct TileCodedFeatures
end




end

