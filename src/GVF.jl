using Lazy
import StatsBase

abstract type AbstractParameterFunction end

function Base.get(apf::AbstractParameterFunction, state_t, action_t, state_tp1, action_tp1, preds_tilde) end

function call(apf::AbstractParameterFunction, state_t, action_t, state_tp1, action_tp1, preds_tilde)
    Base.get(apf::AbstractParameterFunction, state_t, action_t, state_tp1, action_tp1, preds_tilde)
end



"""
Cumulants
"""

abstract type AbstractCumulant <: AbstractParameterFunction end

function Base.get(cumulant::AbstractCumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    throw(DomainError("Base.get(CumulantType, args...) not defined!"))
end


"""
    FeatureCumulant
    - Basic Cumulant which takes the value c_t = s_tp1[idx] for 1<=idx<=length(s_tp1)
"""
struct FeatureCumulant <: AbstractCumulant
    idx::Int
end

Base.get(cumulant::FeatureCumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1) = state_tp1[cumulant.idx]

struct PredictionCumulant <: AbstractCumulant
    idx::Int
end

Base.get(cumulant::PredictionCumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1) = preds_tp1[cumulant.idx]

struct FunctionalCumulant{F} <: AbstractCumulant
    func::F
end

Base.get(cumulant::FunctionalCumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1) =
    cumulant.func(state_t, action_t, state_tp1, action_tp1, preds_tp1)

"""
Discounting
"""
abstract type AbstractDiscount <: AbstractParameterFunction end

function Base.get(γ::AbstractDiscount, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    throw(DomainError("Base.get(DiscountType, args...) not defined!"))
end

struct ConstantDiscount{T} <: AbstractDiscount
    γ::T
end

Base.get(cd::ConstantDiscount, state_t, action_t, state_tp1, action_tp1, preds_tp1) = cd.γ

struct StateTerminationDiscount{T<:Number, F} <: AbstractDiscount
    γ::T
    condition::F
    terminal::T
    StateTerminationDiscount(γ, condition) = new{typeof(γ), typeof(condition)}(γ, condition, convert(typeof(γ), 0.0)) 
end

Base.get(td::StateTerminationDiscount, state_t, action_t, state_tp1, action_tp1, preds_tp1) =
    td.condition(state_t, action_t, state_tp1) ? td.terminal : td.γ


"""
Policies
"""
abstract type AbstractPolicy <: AbstractParameterFunction end

function Base.get(π::AbstractPolicy, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    throw(DomainError("Base.get(PolicyType, args...) not defined!"))
end

StatsBase.sample(rng::Random.AbstractRNG, π::AbstractPolicy, state) = nothing
StatsBase.sample(π::AbstractPolicy, state) = StatsBase.sample(Random.GLOBAL_RNG, π, state)

struct NullPolicy <: AbstractPolicy
end
Base.get(π::NullPolicy, state_t, action_t, state_tp1, action_tp1, preds_tp1) = 1.0

struct PersistentPolicy <: AbstractPolicy
    action::Int64
end

Base.get(π::PersistentPolicy, state_t, action_t, state_tp1, action_tp1, preds_tp1) = π.action == action_t ? 1.0 : 0.0
StatsBase.sample(rng::Random.AbstractRNG, π::PersistentPolicy, state=nothing) = π.action

struct RandomPolicy{T<:AbstractFloat} <: AbstractPolicy
    probabilities::Array{T,1}
    weight_vec::StatsBase.Weights{T, T, Array{T, 1}}
    RandomPolicy(probabilities::Array{T,1}) where {T<:AbstractFloat} = new{T}(probabilities, StatsBase.Weights(probabilities))
end

UniformRandomPolicy(num_actions::Integer) = RandomPolicy(fill(1.0/num_actions, num_actions))
FavoredRandomPolicy(num_actions::Integer, favored_action::Integer, favored_prob::T) where {T<:AbstractFloat}=
    RandomPolicy([ act == favored_action ? favored_prob : (T(1.0) - favored_prob)/(num_actions-1) for act in 1:num_actions])

Base.get(π::RandomPolicy, state_t, action_t, state_tp1, action_tp1, preds_tp1) = π.probabilities[action_t]

StatsBase.sample(rng::Random.AbstractRNG, π::RandomPolicy, state=nothing) = StatsBase.sample(rng, 1:length(π.weight_vec), π.weight_vec)

struct FunctionalPolicy{F} <: AbstractPolicy
    func::F
end

Base.get(π::FunctionalPolicy, state_t, action_t, state_tp1, action_tp1, preds_tp1) =
    π.func(state_t, action_t, state_tp1, action_tp1, preds_tp1)

abstract type AbstractGVF end

function Base.get(gvf::AbstractGVF, state_t, action_t, state_tp1, action_tp1, preds_tp1) end

Base.get(gvf::AbstractGVF, state_t, action_t, state_tp1, preds_tp1) =
    Base.get(gvf::AbstractGVF, state_t, action_t, state_tp1, nothing, preds_tp1)

Base.get(gvf::AbstractGVF, state_t, action_t, state_tp1) =
    Base.get(gvf::AbstractGVF, state_t, action_t, state_tp1, nothing, nothing)

function cumulant(gvf::AbstractGVF) end
function discount(gvf::AbstractGVF) end
function policy(gvf::AbstractGVF) end

struct GVF{C<:AbstractCumulant, D<:AbstractDiscount, P<:AbstractPolicy} <: AbstractGVF
    cumulant::C
    discount::D
    policy::P
end

cumulant(gvf::GVF) = gvf.cumulant
discount(gvf::GVF) = gvf.discount
policy(gvf::GVF) = gvf.policy

function Base.get(gvf::GVF, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    c = Base.get(gvf.cumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    γ = Base.get(gvf.discount, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    π_prob = Base.get(gvf.policy, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    return c, γ, π_prob
end

abstract type AbstractHorde end

struct Horde{T<:AbstractGVF} <: AbstractHorde
    gvfs::Vector{T}
end

function Base.get(gvfh::Horde, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    C = map(gvf -> Base.get(cumulant(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Γ = map(gvf -> Base.get(discount(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Π_probs = map(gvf -> Base.get(policy(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    return C, Γ, Π_probs
end

function Base.get!(C::Array{T, 1}, Γ::Array{T, 1}, Π_probs::Array{T, 1}, gvfh::Horde, state_t, action_t, state_tp1, action_tp1, preds_tp1) where {T<:AbstractFloat}
    C .= map(gvf -> Base.get(cumulant(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Γ .= map(gvf -> Base.get(discount(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Π_probs .= map(gvf -> Base.get(policy(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    return C, Γ, Π_probs
end

Base.get(gvfh::Horde, state_tp1, preds_tp1) = Base.get(gvfh::Horde, nothing, nothing, state_tp1, nothing, preds_tp1)
Base.get(gvfh::Horde, action_t, state_tp1, preds_tp1) = Base.get(gvfh::Horde, nothing, action_t, state_tp1, nothing, preds_tp1)
Base.get(gvfh::Horde, state_t, action_t, state_tp1, preds_tp1) = Base.get(gvfh::Horde, state_t, action_t, state_tp1, nothing, preds_tp1)

Base.get!(C, Γ, Π_probs,gvfh::Horde, action_t, state_tp1, preds_tp1) = Base.get!(C, Γ, Π_probs, gvfh::Horde, nothing, action_t, state_tp1, nothing, preds_tp1)

@forward Horde.gvfs Base.length
