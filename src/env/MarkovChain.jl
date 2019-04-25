# module MarkovChain

# import ..JuliaRL

import Random
import Base.copy

module MarkovChainParams

const RIGHT = 1
const LEFT = 2

const ACTIONS = [RIGHT, LEFT]

end

mutable struct MarkovChain <: AbstractEnvironment
    agent_pos::Int64
    size::Int64
    MarkovChain(size) = new(size/2, size)
    MarkovChain(agent_pos, size) = new(agent_pos, size)
end


JuliaRL.is_terminal(env::MarkovChain) = (env.agent_pos == env.size) || (env.agent_pos == 1)
is_terminal(env::MarkovChain, state) = (state == env.size) || (state == 1)
JuliaRL.get_reward(env::MarkovChain) = if (env.agent_pos == env.size) 1.0 else 0.0 end
get_reward(env::MarkovChain, state) = if (state == env.size) 1.0 else 0.0 end
JuliaRL.get_state(env::MarkovChain) = env.agent_pos
Base.size(env::MarkovChain) = env.size
JuliaRL.get_actions(env::MarkovChain) = MarkovChainParams.ACTIONS
num_actions(env::MarkovChain) = 2
get_states(env::MarkovChain)=collect(1:env.size)

function JuliaRL.reset!(env::MarkovChain; rng=Random.GLOBAL_RNG, kwargs...)
    env.agent_pos = rand(rng, 2:(env.size - 1))
end

function JuliaRL.environment_step!(env::MarkovChain, action; rng=Random.GLOBAL_RNG, kwargs...)
    mcp = MarkovChainParams
    @assert !JuliaRL.is_terminal(env)
    if action == mcp.RIGHT
        env.agent_pos -= 1
    elseif action == mcp.LEFT
        env.agent_pos += 1
    else
        @error "Action invalid $(action)"
    end
end

function _step(env::MarkovChain, action, state; rng=Random.GLOBAL_RNG, kwargs...)
    mcp = MarkovChainParams
    @assert !JuliaRL.is_terminal(env)
    agent_pos = state
    if action == mcp.RIGHT
        agent_pos -= 1
    elseif action == mcp.LEFT
        agent_pos += 1
    else
        @error "Action invalid $(action)"
    end
    return agent_pos, get_reward(env, agent_pos), is_terminal(env, agent_pos)
end

function DynamicProgramming(env::MarkovChain, pi, gamma, thresh=1e-20)
    v = zeros(size(env))
    # v[size(env)] = 1
    delta = 100000
    while thresh < delta
        delta = 0
        for i = 2:9
            _v = v[i]
            r_left = 0
            r_right = if (i == size(env)-1) 1 else 0 end
            v[i] = pi[1]*(r_left + gamma*v[i-1]) + pi[2]*(r_right + gamma*v[i+1])
            delta = max(delta, v[i] - _v)
        end
    end
    return v
end

function Base.show(io::IO, env::MarkovChain)
    println("MARKOV CHAIN: $(get_state(env)), $(get_reward(env)), $(is_terminal(env))")
    model = fill("0", env.size)
    model[1] = "1"
    println(prod(model.*" "))
    model = fill("-", env.size)
    model[env.agent_pos] = "a"
    println(prod(model.*" "))
end

