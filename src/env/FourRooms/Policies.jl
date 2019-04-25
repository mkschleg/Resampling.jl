

# export random_behaviour_policy

# include("../../GVF.jl")
using StatsBase

function create_persistant_policy(action)
    next_action_pi(state_t; rng=Random.GLOBAL_RNG) = 1
    prob_pi(state_t, action_t) = if (action_t == action) 1 else 0 end
    pi = GeneralValueFunction.Policy(prob_pi, next_action_pi)
    return pi
end


random_behaviour_next_action(state_t; rng=Random.GLOBAL_RNG) = rand(rng, 1:4)
random_behaviour_probability(state_t, action_t) = 0.25
random_behaviour_policy = GeneralValueFunction.Policy(
    random_behaviour_probability, random_behaviour_next_action)

var_random_behaviour_next_action(state_t; rng=Random.GLOBAL_RNG) = StatsBase.sample(rng, 1:4, Weights([0.9/3, 0.9/3, 0.1, 0.9/3]))
var_random_behaviour_probability(state_t, action_t) = if (action_t == 3) 0.1 else 0.9/3 end
var_random_behaviour_policy = GeneralValueFunction.Policy(
    var_random_behaviour_probability, var_random_behaviour_next_action)


function next_action_variant(state_t; rng=Random.GLOBAL_RNG)
    high_prob = 0.45
    low_prob = 0.05
    # UP, RIGHT, DOWN, LEFT
    if state_t == [3,2]
        # Left Top
        act = StatsBase.sample(rng, 1:4, Weights([high_prob, low_prob, low_prob, high_prob]))
    elseif state_t == [3,9]
        # Right Top
        act = StatsBase.sample(rng, 1:4, Weights([high_prob, high_prob, low_prob, low_prob]))
    elseif state_t == [10,2]
        # Left Down
        act = StatsBase.sample(rng, 1:4, Weights([low_prob, low_prob, high_prob, high_prob]))
    elseif state_t == [10,9]
        # Right Down
        act = StatsBase.sample(rng, 1:4, Weights([low_prob, high_prob, high_prob, low_prob]))
    else
        act = rand(rng, 1:4)
    end
    return act
end

function probability_variant(state_t, action_t)
    high_prob = 0.45
    low_prob = 0.05
    # UP, RIGHT, DOWN, LEFT
    if state_t == [3,2]
        # Left Top
        if action_t == 1 || action_t == 4
            return high_prob
        else
            return low_prob
        end
    elseif state_t == [3,9]
        # Right Top
        if action_t == 1 || action_t == 2
            return high_prob
        else
            return low_prob
        end
    elseif state_t == [10,2]
        # Left Down
        if action_t == 3 || action_t == 4
            return high_prob
        else
            return low_prob
        end
    elseif state_t == [10,9]
        # Right Down
        if action_t == 2 || action_t == 3
            return high_prob
        else
            return low_prob
        end
    else
        return 0.25
    end
    return act
end

variant_behaviour_policy = GeneralValueFunction.Policy(
    probability_variant, next_action_variant)

function next_action_more_variant(state_t; rng=Random.GLOBAL_RNG)
    high_prob = 0.95/3
    low_prob = 0.05
    # UP, RIGHT, DOWN, LEFT
    # Assume target policy is going down.
    # Generated with gen_states(1143139448, 25)
    states = [[6, 7], [4, 8], [6, 9], [3, 11], [2, 10],
              [6, 2], [8, 1], [5, 2], [11, 8], [1, 2],
              [10, 1], [1, 5], [4, 7], [10, 11], [2, 9],
              [6, 5], [9, 4], [3, 6], [4, 1], [3, 2],
              [7, 5], [3, 9], [5, 6], [10, 9], [2, 1]]

    if state_t in states
        act = StatsBase.sample(rng, 1:4, Weights([high_prob, high_prob, low_prob, high_prob]))
    else
        act = rand(rng, 1:4)
    end
    return act
end

function gen_states(seed, num)
    mt = Random.MersenneTwister(seed)
    states = []
    while length(states) < num
        new_state = rand(mt, 1:11, 2)
        if !(new_state in states)
            append!(states, [new_state])
        end
    end
    return states
end

function find_same(states)
    s=0
    for p in collect(Iterators.product(states,states))
        s += if (p[1] == p[2]) 1 else 0 end
    end
    return (s - length(states))/2
end


function probability_more_variant(state_t, action_t)
    high_prob = 0.95/3
    low_prob = 0.05
    # UP, RIGHT, DOWN, LEFT
    # Assume target policy is going Down!
    # Generated with gen_states(1143139448, 25)
    states = [[6, 7], [4, 8], [6, 9], [3, 11], [2, 10],
              [6, 2], [8, 1], [5, 2], [11, 8], [1, 2],
              [10, 1], [1, 5], [4, 7], [10, 11], [2, 9],
              [6, 5], [9, 4], [3, 6], [4, 1], [3, 2],
              [7, 5], [3, 9], [5, 6], [10, 9], [2, 1]]
    
    if state_t in states
        # Assume
        if action_t != 3
            return high_prob
        else
            return low_prob
        end
    else
        return 0.25
    end
    return act
end

more_variant_behaviour_policy = GeneralValueFunction.Policy(
    probability_more_variant, next_action_more_variant)



function next_action_super_variant(state_t; rng=Random.GLOBAL_RNG)
    high_prob = 0.95/3
    low_prob = 0.05
    # UP, RIGHT, DOWN, LEFT
    # Assume target policy is going down.
    # Generated with gen_states(1143139448, 25)
    # states = [[6, 7], [4, 8], [6, 9], [3, 11], [2, 10],
    #           [6, 2], [8, 1], [5, 2], [11, 8], [1, 2],
    #           [10, 1], [1, 5], [4, 7], [10, 11], [2, 9],
    #           [6, 5], [9, 4], [3, 6], [4, 1], [3, 2],
    #           [7, 5], [3, 9], [5, 6], [10, 9], [2, 1]]
    states = [[6, 7], [4, 8], [6, 9], [3, 11], [2, 10],
              [6, 2], [8, 1], [5, 2], [11, 8], [1, 2],
              [10, 1], [1, 5], [4, 7], [10, 11], [2, 9],
              [6, 5], [9, 4], [3, 6], [4, 1], [3, 2],
              [7, 5], [3, 9], [5, 6], [10, 9], [2, 1],
              [6, 3], [8, 11], [9, 5], [11, 3], [8, 2],
              [5, 9], [7, 11], [10, 10], [9, 8], [1, 10],
              [7, 9], [1, 7], [3, 1], [7, 10], [7, 7],
              [10, 2], [10, 8], [1, 9], [11, 9], [1, 6],
              [11, 10], [8, 8], [10, 5], [5, 11], [2, 2]]
    if state_t in states
        act = StatsBase.sample(rng, 1:4, Weights([high_prob, high_prob, low_prob, high_prob]))
    else
        act = rand(rng, 1:4)
    end
    return act
end

function probability_super_variant(state_t, action_t)
    high_prob = 0.95/3
    low_prob = 0.05
    # UP, RIGHT, DOWN, LEFT
    # Assume target policy is going Down!
    # Generated with gen_states(1143139448, 25)
    # states = [[6, 7], [4, 8], [6, 9], [3, 11], [2, 10],
    #           [6, 2], [8, 1], [5, 2], [11, 8], [1, 2],
    #           [10, 1], [1, 5], [4, 7], [10, 11], [2, 9],
    #           [6, 5], [9, 4], [3, 6], [4, 1], [3, 2],
    #           [7, 5], [3, 9], [5, 6], [10, 9], [2, 1]]
    states = [[6, 7], [4, 8], [6, 9], [3, 11], [2, 10],
              [6, 2], [8, 1], [5, 2], [11, 8], [1, 2],
              [10, 1], [1, 5], [4, 7], [10, 11], [2, 9],
              [6, 5], [9, 4], [3, 6], [4, 1], [3, 2],
              [7, 5], [3, 9], [5, 6], [10, 9], [2, 1],
              [6, 3], [8, 11], [9, 5], [11, 3], [8, 2],
              [5, 9], [7, 11], [10, 10], [9, 8], [1, 10],
              [7, 9], [1, 7], [3, 1], [7, 10], [7, 7],
              [10, 2], [10, 8], [1, 9], [11, 9], [1, 6],
              [11, 10], [8, 8], [10, 5], [5, 11], [2, 2]]
    if state_t in states
        # Assume
        if action_t != 3
            return high_prob
        else
            return low_prob
        end
    else
        return 0.25
    end
    return act
end

super_variant_behaviour_policy = GeneralValueFunction.Policy(
    probability_super_variant, next_action_super_variant)

function next_action_rand_more_variant(states, state_t; rng=Random.GLOBAL_RNG)
    high_prob = 0.95/3
    low_prob = 0.05

    if state_t in states
        act = StatsBase.sample(rng, 1:4, Weights([high_prob, high_prob, low_prob, high_prob]))
    else
        act = rand(rng, 1:4)
    end
    return act
end

function probability_rand_more_variant(states, state_t, action_t)
    high_prob = 0.95/3
    low_prob = 0.05

    if state_t in states
        # Assume
        if action_t != 3
            return high_prob
        else
            return low_prob
        end
    else
        return 0.25
    end
    return act
end



mutable struct super_random
    dist
    function super_random(env_shape, num_actions; rng=Random.GLOBAL_RNG, ϵ=0.05)
        d = rand(rng, num_actions, env_shape[1], env_shape[2])
        d = clamp.(d, ϵ, 1)
        d ./= sum(d;dims=1)
        new(d)
    end
end

function next_action_random(strg::super_random, state_t; rng=Random.GLOBAL_RNG)
    high_prob = 0.95/3
    low_prob = 0.05
    # UP, RIGHT, DOWN, LEFT
    state_prob = strg.dist[:, state_t[1], state_t[2]]
    a = rand(rng)
    act = -1
    act_sum = 0
    for i in 1:length(state_prob)
        act_sum += state_prob[i]
        if act_sum < a
            return i
        end
    end
end

function probability_random(strg::super_random, state_t, action_t)
    state_prob = strg.dist[:, state_t[1], state_t[2]]
    return state_prob[action_t]
end


function create_random(env_size, num_actions; rng=Random.GLOBAL_RNG, strg=nothing)
    if strg == nothing
        strg = super_random(env_size, num_actions; rng=rng)
    end
    p(state_t, action_t) = probability_random(strg, state_t, action_t)
    act(state_t; rnng=rng) = next_action_random(strg, state_t; rng=rnng)
    return GeneralValueFunction(p, act)
end

# super_variant_behaviour_policy = GeneralValueFunction.Policy(
#     probability_super_variant, next_action_super_variant)





# Changing behaviour policy

mutable struct changing_behavior
    freq
    step
end

function next_action_cp(CB::changing_behavior, state_t; rng=Random.GLOBAL_RNG)
    
end

function probability_cp(CB::changing_behavior, state_t, action_t; rng=Random.GLOBAL_RNG)

end


function create_changing_behavior_policy(freq)
    changing_behavior(freq, 0)

end


