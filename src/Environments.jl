# module Environments

# abstract type EnvironmentState end

# is_terminal(env::EnvironmentState) = (env.agent_pos == env.size) || (env.agent_pos == 1)
# reward(env::EnvironmentState) = if (env.agent_pos == env.size) 1 else 0 end
# get_state(env::EnvironmentState) = [env.agent_pos]
# size(env::EnvironmentState) = env.size
# size_action(env::EnvironmentState) = 2
# actions(env::EnvironmentState) = ACTIONS

# function start!(env::EnvironmentState; rng=Random.GLOBAL_RNG)
#     throw(ErrorException("Implement start! for $(typeof(opt))"))
# end

# function step(env::EnvironmentState, action)
#     new_env = copy(env)
#     return step!(new_env, action)
# end

# function step!(env::EnvironmentState, action)
#     throw(ErrorException("Implement step! for $(typeof(opt))"))
# end


export FourRooms, MarkovChain, StarProblem

export step!, start!, get_actions, is_terminal, get_reward, get_state
import JuliaRL: AbstractEnvironment, step!, start!, get_actions, is_terminal, get_reward, get_state

include("env/FourRooms.jl")
include("env/MarkovChain.jl")
# include("env/StarProblem.jl")


# end

