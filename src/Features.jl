module Features
import Combinatorics: combinations

import JuliaRL.FeatureCreators


struct RandomFeatures{T, F<:AbstractFloat} <: FeatureCreators.AbstractFeatureCreator
    # feature_map::Array{Array{Float64, 1}, 1}
    feature_map::Dict{T, Array{F, 1}}
    size::Int64
    active::Int64
    function RandomFeatures{F}(state_list::Array{T, 1}, feature_size, active_features; rng=Random.GLOBAL_RNG) where {T, F<:AbstractFloat}
        comb = shuffle(rng, collect(combinations(1:feature_size, active_features)))
        features = [zeros(F, feature_size) for i in 1:chain_size]
        for (idx, state) in enumerate(state_list)
            features[state[comb[idx]]] .= 1.0f0
        end
        new(features, feature_size, active_features)
    end
end

@forward RandomFeatures.features Base.getindex

FeatureCreators.create_features(fc::RandomFeatures, s; kwargs...) = fc[s]
FeatureCreators.feature_size(fc::RandomFeatures) = fc.size

end

