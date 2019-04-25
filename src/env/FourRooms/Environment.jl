import Random, Base.size

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

ROOM_TOP_LEFT = 1
ROOM_TOP_RIGHT = 2
ROOM_BOTTOM_LEFT = 3
ROOM_BOTTOM_RIGHT = 4

mutable struct environment
    state
    walls::Array{Bool, 2}
    environment() = new([0,0], convert(Array{Bool}, BASE_WALLS))
    environment(walls::Array{Int64, 2}) = new([0,0], convert(Array{Bool}, walls))
    function environment(size, wall_list)
        walls = zeros(Bool, size[1], size[2])
        for wall in wall_list
            walls[wall[1], wall[2]] = true
        end
        new((0,0), walls)
    end
end

is_terminal(env::environment) = false
reward(env::environment) = 0
state(env::environment) = env.state
is_wall(env::environment, state) = env.walls[state[1], state[2]] == 1
random_state(env::environment, rng) = [rand(rng, 1:size(env.walls)[1]), rand(rng, 1:size(env.walls)[2])]
size(env::environment) = size(env.walls)

function which_room(state)
    room = -1
    if state[1] < 6
        # LEFT
        if state[2] < 6
            # TOP
            room = ROOM_TOP_LEFT
        else
            # Bottom
            room = ROOM_BOTTOM_LEFT
        end
    else
        # RIGHT
        if state[2] < 7
            # TOP
            room = ROOM_TOP_RIGHT
        else
            # Bottom
            room = ROOM_BOTTOM_RIGHT
        end
    end
    return room
end

function start!(env::environment; rng=Random.GLOBAL_RNG)
    # state = random_state(rng)
    # state = rand(rng, [1:size(env.walls)[1], 1:size(env.walls)[2]])
    state = random_state(env, rng)
    while !env.walls[state[1], state[2]]
        state = random_state(env, rng)
    end
    env.state = state
    return state
end

function step!(env::environment, action)
    (env.state, rew, terminal) = step(env, action)
    return env.state, rew, terminal
end

function step(env::environment, action)
    # print("hello world")
    next_state = copy(env.state)

    if action == UP
        next_state[1] -= 1
    elseif action == DOWN
        next_state[1] += 1
    elseif action == RIGHT
        next_state[2] += 1
    elseif action == LEFT
        next_state[2] -= 1
    end

    next_state[1] = clamp(next_state[1], 1, size(env.walls)[1])
    next_state[2] = clamp(next_state[2], 1, size(env.walls)[2])
    if is_wall(env, next_state)
        next_state = env.state
    end

    return next_state, 0, false
end

function step(env::environment, state, action)
    # print("hello world")
    next_state = copy(state)

    if action == UP
        next_state[1] -= 1
    elseif action == DOWN
        next_state[1] += 1
    elseif action == RIGHT
        next_state[2] += 1
    elseif action == LEFT
        next_state[2] -= 1
    end

    next_state[1] = clamp(next_state[1], 1, size(env.walls)[1])
    next_state[2] = clamp(next_state[2], 1, size(env.walls)[2])
    if is_wall(env, next_state)
        next_state = state
    end

    return next_state, 0, false
end
