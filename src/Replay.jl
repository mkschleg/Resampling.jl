
include("Buffer.jl")
include("SumTree.jl")

import Random
import Base.getindex, Base.size

"""
AbstractExperienceReplay

    Abstract notion of an experience replay buffer for deep reinforcement learning.

"""
abstract type AbstractExperienceReplay end

size(er::AbstractExperienceReplay) = size(er.buffer)
getindex(er::AbstractExperienceReplay, idx) = getindex(er.buffer, idx)
add!(er::AbstractExperienceReplay, experience, args...) = add!(er.buffer, experience)

sample(er::AbstractExperienceReplay, batch_size; rng=Random.GLOBAL_RNG) = nothing


"""
ExperienceReplay
"""
mutable struct ExperienceReplay <: AbstractExperienceReplay
    """Experience Buffer"""
    buffer::CircularBuffer
    ExperienceReplay(size, types; column_names=nothing)= new(
        CircularBuffer(size, types; column_names = column_names))
end

size(er::ExperienceReplay) = size(er.buffer)
getindex(er::ExperienceReplay, idx) = getindex(er.buffer, idx)
add!(er::ExperienceReplay, experience, args...) = add!(er.buffer, experience)

"""
    sample(er, batch_size[; rng])

    Sample from the experience replay buffer a batch of size (batch_size) according to the random number generator (rng).

"""
function sample(er::ExperienceReplay, batch_size; rng=Random.GLOBAL_RNG)
    idx = rand(rng, 1:size(er)[1], batch_size)
    return getrow(er.buffer, idx)
end

"""
    WeightedExperienceReplay

    Samples according to generic weights passed in during experience addition.

"""
mutable struct WeightedExperienceReplay <: AbstractExperienceReplay
    """Experience Buffer"""
    buffer::CircularBuffer
    """managing the weights of the components"""
    sumtree::SumTree
    WeightedExperienceReplay(size, types; column_names=nothing)= new(
        CircularBuffer(size, types; column_names = column_names),
        SumTree{Int64}(size))
end

size(er::WeightedExperienceReplay) = size(er.buffer)
getindex(er::WeightedExperienceReplay, idx) = getindex(er.buffer, idx)
total(er::WeightedExperienceReplay) = total(er.sumtree)
add!(er::WeightedExperienceReplay, experience, args...) = add!(er, experience, args[1])
function add!(er::WeightedExperienceReplay, experience, weight, args...)
    idx = add!(er.buffer, experience)
    add!(er.sumtree, weight, idx)
    return
end

function sample(er::WeightedExperienceReplay, batch_size; rng=Random.GLOBAL_RNG)
    batch_idx, batch_priorities, idx = sample(er.sumtree, batch_size; rng=rng)
    return getrow(er.buffer, idx)
end






