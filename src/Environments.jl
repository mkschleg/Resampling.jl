

export FourRooms, FourRoomsCont, MarkovChain, StarProblem

export step!, start!, get_actions, is_terminal, get_reward, get_state
import JuliaRL: AbstractEnvironment, step!, start!, get_actions, is_terminal, get_reward, get_state

include("env/FourRooms.jl")
include("env/FourRoomsCont.jl")
include("env/MarkovChain.jl")
