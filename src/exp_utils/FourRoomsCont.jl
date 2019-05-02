module FourRoomsContUtil

using ..Resampling
import ..Resampling.FourRoomsParams
import ..Resampling: AbstractPolicy
using ..Reproduce
import StatsBase
import StatsBase: Weights
using Random
import ..JuliaRL
import ..build_algorithm_dict

include("FourRoomsCont/policies.jl")

export
    TCFourRoomsContAgent, NNFourRoomsContAgent,
    predict, predict!

include("FourRoomsCont/agent.jl")
include("FourRoomsCont/nn_agent.jl")

include("FourRoomsCont/dmu.jl")

export
    env_settings!,
    get_experience,
    policy_settings,
    max_is_dict,
    get_policy

const constructed_policy_settings = Dict(
    "uniform"=>UniformRandomPolicy(4),
    "corners"=>FourCornersVariant(),
    "favor_down"=>FavoredRandomPolicy(4, FourRoomsParams.DOWN, 0.1),
    "random_state_variant"=>RandomStateVariant(),
    "random_state_weight_variant"=>RandomStateWeightVariant()
)

const GVFS = Dict(
    "collide_down"=>GVF(FunctionalCumulant((state_t, action_t, state_tp1, action_tp1, preds_tp1)-> (state_tp1[2] ? 1.0 : 0.0)),
                StateTerminationDiscount(0.9, (state_t, action_t, state_tp1)->(state_tp1[2])),
                FavoredRandomPolicy(4, 3, 1.0)),
    "favored_down"=>GVF(FunctionalCumulant((state_t, action_t, state_tp1, action_tp1, preds_tp1)-> (state_tp1[2] ? 1.0 : 0.0)),
                        StateTerminationDiscount(0.9, (state_t, action_t, state_tp1)->(state_tp1[2])),
                        FavoredRandomPolicy(4, 3, 0.9))
)

get_max_is_ratio(μ::RandomPolicy, π::RandomPolicy) =
    maximum(π.probabilities./μ.probabilites)
get_max_is_ratio(μ::FRStateDependentPolicy, π::RandomPolicy) =
    maximum([maximum(π.probabilities./μ.weights[state]) for state in μ.states])


function env_settings!(s::ArgParseSettings)
    @add_arg_table s begin
        "--policy"
        help = "The policy settings: $(keys(constructed_policy_settings))"
        range_tester=(p)->(p ∈ keys(constructed_policy_settings))
        required=true
        "--gvf"
        help = "The gvf to learn: $(keys(GVFS))"
        range_tester=(gvf)->(gvf ∈ keys(GVFS))
        required=true
    end
end

function get_policy(parsed::Dict)
    return constructed_policy_settings[parsed["policy"]]
end

import ProgressMeter

function get_dmu_mu_dist(env::FourRoomsCont, policy; seed=0, num_runs=1e5, num_steps=1000)
    rng = Random.MersenneTwister(seed)

    heat_map = zeros(size(env)...)
    
    ProgressMeter.@showprogress 0.1 "Run:" for i in 1:num_runs
        _, state = start!(env; rng=rng)
        action = StatsBase.sample(rng, policy, state[1])
        for i in 1:num_steps
            _, state, _, _ = step!(env, action; rng=rng)
            action = StatsBase.sample(rng, policy, state[1])
        end
        heat_map[Resampling.project(env, state[1])...] += 1
    end
    return heat_map./sum(heat_map)
end


function sample_according_to_dmu(env::FourRoomsCont, policy_str::AbstractString, num_states; rng=Random.GLOBAL_RNG)
    dmu = mu_dmu_dict[policy_str]
    indicies = reshape(CartesianIndices(dmu), :)
    w = StatsBase.Weights(dmu[indicies])

    prj_states = StatsBase.sample(rng, indicies, w, num_states)

    [[rand(rng) + (s[1]-1), rand(rng) + (s[2]-1)] for s in prj_states]
end

end

