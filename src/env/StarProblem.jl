module StarProblem

import Random
import Base.copy, Base.size

ACTION_1 = 0
ACTION_2 = 1

PHI = [[1,2,0,0,0,0,0,0], # state 1
       [1,0,2,0,0,0,0,0], # state 2
       [1,0,0,2,0,0,0,0], # state 3
       [1,0,0,0,2,0,0,0], # state 4
       [1,0,0,0,0,2,0,0], # state 5
       [1,0,0,0,0,0,2,0], # state 6
       [2,0,0,0,0,0,0,1]] # state 7

INITIAL = [1,1,1,1,1,1,1,10]

mutable struct environment
    agent_pos::Int64
    environment() = new(rand(1:6))
end

is_terminal(env::environment) = false
reward(env::environment) = 0
get_state(env::environment) = PHI[env.agent_pos]
size(env::environment) = 8

function start!(env; rng=Random.GLOBAL_RNG)
    env.agent_pos = rand(rng, 1:6)
    return get_state(env)
end

function step(env::environment, action; rng=Random.GLOBAL_RNG)
    new_env = environment(env.agent_pos, env.size; rng=rng)
    return step!(new_env, action)
end

function step!(env::environment, action; rng=Random.GLOBAL_RNG)
    @assert !is_terminal(env)
    if action == ACTION_1
        # env.agent_pos -= 1
        env.agent_pos = 7
    elseif action == ACTION_2
        env.agent_pos = rand(rng, 1:6)
    else
        @error "Action invalid"
    end
    return get_state(env), reward(env), is_terminal(env)
end

TRUTH = [0]

end
