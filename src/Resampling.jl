module Resampling

import JuliaRL
import Base.get

glorot_uniform(rng::Random.AbstractRNG, dims...) = begin; (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims)) end;
glorot_normal(rng::Random.AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))


export
    AbstractExperienceReplay,
    ExperienceReplay,
    WeightedExperienceReplay,
    add!,
    size,
    getindex,
    sample

include("Replay.jl")


export
    GVF,
    FunctionalCumulant,
    StateTerminationDiscount,
    ConstantDiscount,
    PersistentPolicy,
    RandomPolicy,
    UniformRandomPolicy,
    FavoredRandomPolicy

include("GVF.jl")

export Tabular, SparseLayer
include("CustomLayers.jl")

export BatchTD, WISBatchTD, VTrace, BatchSarsa, BatchExpectedSarsa, update!
include("Learning.jl")


# include("LinearRL.jl")
# include("TabularRL.jl")

include("Environments.jl")

export MarkovChainUtil
include("exp_util.jl")







end # module
