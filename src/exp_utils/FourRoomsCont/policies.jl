

struct FRStateDependentPolicy{S, F<:AbstractFloat} <: AbstractPolicy
    states::Array{S, 1}
    weights::Dict{S, Weights{F, F, Array{F, 1}}}
end


function Base.get(π::FRStateDependentPolicy, state_t::Array{Float64, 1}, action_t, state_tp1, action_tp1, preds_tp1)
    prob = -1.0
    prj = Int64.(floor.(state_t) .+ 1)
    if prj ∈ π.states
        prob = π.weights[prj][action_t]
    else
        prob = 0.25
    end
    return prob
end


function StatsBase.sample(rng::Random.AbstractRNG, π::FRStateDependentPolicy, state::Array{Float64, 1})
    act = -1
    prj = Int64.(floor.(state) .+ 1)
    if prj ∈ π.states
        act = StatsBase.sample(rng, 1:4, π.weights[prj])
    else
        act = rand(rng, 1:4)
    end
    return act
end

function FourCornersVariant(high_prob=0.45, low_prob=0.05)
    map = Dict(
        [3,2]=>Weights([high_prob, low_prob, low_prob, high_prob]),
        [3,9]=>Weights([high_prob, high_prob, low_prob, low_prob]),
        [10,2]=>Weights([low_prob, low_prob, high_prob, high_prob]),
        [10,9]=>Weights([low_prob, high_prob, high_prob, low_prob])
    )
    states = collect(keys(map))
    return FRStateDependentPolicy(states, map)
end


function find_same(states)
    s=0
    for p in collect(Iterators.product(states,states))
        s += if (p[1] == p[2]) 1 else 0 end
    end
    return (s - length(states))/2
end

# function gen_states(seed, num)
#     mt = Random.MersenneTwister(seed)
#     states = []
#     while length(states) < num
#         new_state = rand(mt, 1:11, 2)
#         if !(new_state in states)
#             append!(states, [new_state])
#         end
#     end
#     return states
# end

function gen_states(rng::Random.AbstractRNG, num)
    states = Array{Int64, 1}[]
    while length(states) < num
        new_state = rand(rng, 1:11, 2)
        if !(new_state in states)
            append!(states, [new_state])
        end
    end
    return states
end

gen_states(seed::Int64, num) = gen_states(Random.MersenneTwister(seed), num_states)


function RandomStateVariant(action, prob, num_states, seed::Int64)
    states = gen_states(Random.MersenneTwister(seed), num_states)
    w = Weights([act == action ? prob : (1.0-prob)/3
                 for act in 1:4])
    map = Dict([s=>w for s in states])
    return FRStateDependentPolicy(states, map)
end

function RandomStateVariant(action, prob, num_states, rng::Random.AbstractRNG)
    states = gen_states(rng, num_states)
    w = Weights([act == action ? prob : (1.0-prob)/3
                 for act in 1:4])
    map = Dict([s=>w for s in states])
    return FRStateDependentPolicy(states, map)
end

RandomStateVariant() =
    RandomStateVariant(FourRoomsParams.DOWN, 0.05, 25, 1143139448)

function random_weight_vector(rng::Random.AbstractRNG, num_actions)
    favored_action = rand(rng, 1:num_actions)
    prob = 0.5*(rand(rng) + 1.0)
    [act == favored_action ? prob : (1.0-prob)/(num_actions-1) for act in 1:num_actions]
end

function RandomStateWeightVariant(num_states::Int64, seed::Int64)
    rng = Random.MersenneTwister(seed)
    states = gen_states(rng, num_states)
    weights = Dict([state=>Weights(random_weight_vector(rng, 4)) for state in states])
    FRStateDependentPolicy(states, weights)
end

RandomStateWeightVariant() =
    RandomStateWeightVariant(25, 1143139448)

