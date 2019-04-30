# module FourRooms

import Random, Base.size
# import IWER.GeneralValueFunction


module FourRoomsParams

BASE_WALLS = [0 0 0 0 0 1 0 0 0 0 0;
              0 0 0 0 0 1 0 0 0 0 0;
              0 0 0 0 0 0 0 0 0 0 0;
              0 0 0 0 0 1 0 0 0 0 0;
              0 0 0 0 0 1 0 0 0 0 0;
              1 0 1 1 1 1 0 0 0 0 0;
              0 0 0 0 0 1 1 1 0 1 1;
              0 0 0 0 0 1 0 0 0 0 0;
              0 0 0 0 0 1 0 0 0 0 0;
              0 0 0 0 0 0 0 0 0 0 0;
              0 0 0 0 0 1 0 0 0 0 0;]

UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4

ACTIONS = [UP, RIGHT, DOWN, LEFT]

ROOM_TOP_LEFT = 1
ROOM_TOP_RIGHT = 2
ROOM_BOTTOM_LEFT = 3
ROOM_BOTTOM_RIGHT = 4

end


"""
    FourRooms

    Four Rooms environment using JuliaRL abstract environment.

    - state: [y, x]


"""
mutable struct FourRooms <: AbstractEnvironment
    state::Array{Int64, 1}
    walls::Array{Bool, 2}
end

FourRooms() = FourRooms([0,0], convert(Array{Bool}, FourRoomsParams.BASE_WALLS))
FourRooms(walls::Array{Int64, 2}) = FourRooms([0,0], convert(Array{Bool}, walls))
function FourRooms(size::Int, wall_list::Array{CartesianIndex{2}})
    walls = fill(false, size[1], size[2])
    for wall in wall_list
        walls[wall] = true
    end
    FourRooms((0,0), walls)
end

JuliaRL.is_terminal(env::FourRooms) = false
JuliaRL.get_reward(env::FourRooms) = 0
JuliaRL.get_state(env::FourRooms) = env.state
is_wall(env::FourRooms, state) = env.walls[state[1], state[2]] == 1
random_state(env::FourRooms, rng) = [rand(rng, 1:size(env.walls)[1]), rand(rng, 1:size(env.walls)[2])]
Base.size(env::FourRooms) = size(env.walls)
num_actions(env::FourRooms) = 4
get_states(env::FourRooms) = findall(x->x==false, env.walls)
JuliaRL.get_actions(env::FourRooms) = FourRoomsParams.ACTIONS

function which_room(env::FourRooms, state)
    frp = FourRoomsParams
    room = -1
    if state[1] < 6
        # LEFT
        if state[2] < 6
            # TOP
            room = frp.ROOM_TOP_LEFT
        else
            # Bottom
            room = frp.ROOM_BOTTOM_LEFT
        end
    else
        # RIGHT
        if state[2] < 7
            # TOP
            room = frp.ROOM_TOP_RIGHT
        else
            # Bottom
            room = frp.ROOM_BOTTOM_RIGHT
        end
    end
    return room
end

function JuliaRL.reset!(env::FourRooms; rng=Random.GLOBAL_RNG, kwargs...)
    state = random_state(env, rng)
    while env.walls[state[1], state[2]]
        state = random_state(env, rng)
    end
    env.state = state
    return state
end

function JuliaRL.environment_step!(env::FourRooms, action; rng=Random.GLOBAL_RNG, kwargs...)


    frp = FourRoomsParams
    next_state = copy(env.state)

    if action == frp.UP
        next_state[1] -= 1
    elseif action == frp.DOWN
        next_state[1] += 1
    elseif action == frp.RIGHT
        next_state[2] += 1
    elseif action == frp.LEFT
        next_state[2] -= 1
    end

    next_state[1] = clamp(next_state[1], 1, size(env.walls)[1])
    next_state[2] = clamp(next_state[2], 1, size(env.walls)[2])
    if is_wall(env, next_state)
        next_state = env.state
    end

    env.state = next_state
end

function _step(env::FourRooms, state, action; rng=Random.GLOBAL_RNG, kwargs...)
    frp = FourRoomsParams
    next_state = copy(state)

    if action == frp.UP
        next_state[1] -= 1
    elseif action == frp.DOWN
        next_state[1] += 1
    elseif action == frp.RIGHT
        next_state[2] += 1
    elseif action == frp.LEFT
        next_state[2] -= 1
    end

    next_state[1] = clamp(next_state[1], 1, size(env.walls)[1])
    next_state[2] = clamp(next_state[2], 1, size(env.walls)[2])
    if is_wall(env, next_state)
        next_state = state
    end

    return next_state, 0, false
end

function _step(env::FourRooms, state::CartesianIndex{2}, action)
    array_state = [state[1], state[2]]
    new_state, r, t = _step(env, array_state, action)
    return CartesianIndex{2}(new_state[1], new_state[2]), r, t

end


function DynamicProgramming(env::FourRooms, gvf::GVF, thresh=1e-20)
    v = zeros(size(env))
    states = get_states(env)
    delta = 100000
    while thresh < delta
        delta = 0
        for state in states
            _v = v[state]
            v_new = 0.0
            for action in JuliaRL.get_actions(env)
                new_state, _, _ = _step(env, state, action)
                value_act = v[new_state]
                c, γ, π = get(gvf, state, action, new_state)
                v_new += π*(c + γ*value_act)
            end
            v[state] = v_new
            delta = max(delta, abs(_v - v[state]))
        end
    end
    return v
end
