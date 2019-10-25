module Resampling

import JuliaRL
import Base.get

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
    InterpolationPolicy,
    FavoredRandomPolicy

include("GVF.jl")

export Tabular, SparseLayer
include("CustomLayers.jl")

export
    BatchTD,
    IncTD,
    WISBatchTD,
    VTrace,
    IncNormIS,
    WSNormIS,
    BatchSarsa,
    BatchExpectedSarsa,
    update!
include("Learning.jl")

include("Environments.jl")

export MarkovChainUtil
include("exp_util.jl")



# Defunc learning algorithms ported from old codebase for the variance experiments:

include("TabularRL.jl")





end # module
