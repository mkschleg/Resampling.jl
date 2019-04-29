module FourRoomsUtil

using ..Resampling
import ..Resampling.FourRoomsParams
import ..Resampling: AbstractPolicy
using ..Reproduce
import StatsBase
import StatsBase: Weights
using Random
import ..JuliaRL
import ..build_algorithm_dict

include("FourRooms/policies.jl")

export FourRoomsAgent

include("FourRooms/agent.jl")

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
    "random_state_variant"=>RandomStateVariant()
)

const GVFS = Dict(
    "down"=>GVF(FunctionalCumulant((state_t, action_t, state_tp1, action_tp1, preds_tp1)-> (state_tp1 == state_t ? 1.0 : 0.0)),
                StateTerminationDiscount(0.9, (state_t, action_t, state_tp1)->(state_tp1 == state_t)),
                PersistentPolicy(3)),
    "favored_down"=>GVF(FunctionalCumulant((state_t, action_t, state_tp1, action_tp1, preds_tp1)-> (state_tp1 == state_t ? 1.0 : 0.0)),
                        StateTerminationDiscount(0.9, (state_t, action_t, state_tp1)->(state_tp1 == state_t)),
                        FavoredRandomPolicy(4, 3, 0.9))
)


const max_is_dict = Dict(
    "easy"=>1.8,
    "hard"=>9.0,
    "hardest"=>99.0)

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
        # "--policy_input"
        # help="Inputs to the policy"
        # nargs='+'
    end
end

# function get_policy(policy)
#     return policy_settings[policy][1], policy_settings[policy][2]
# end

function get_policy(parsed::Dict)
    return constructed_policy_settings[parsed["policy"]]
end



end

