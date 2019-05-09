module MarkovChainUtil

using ..Resampling
using ..Reproduce
import StatsBase

export
    env_settings!,
    get_experience,
    get_experience_interpolation,
    policy_settings,
    max_is_dict,
    get_policies

const policy_settings = Dict(
    "easy"=>([0.5, 0.5], [0.1, 0.9]),
    "hard"=>([0.9, 0.1], [0.1, 0.9]),
    "hardest"=>([0.99, 0.01], [0.01, 0.99]))

const max_is_dict = Dict(
    "easy"=>1.8,
    "hard"=>9.0,
    "hardest"=>99.0)

function env_settings!(s::ArgParseSettings)
    @add_arg_table s begin
        "--chainsize"
        arg_type=Int64
        default=10
        "--policy"
        help = "The policy settings"
        default="easy"
    end
end

function get_policies(policy)
    return policy_settings[policy][1], policy_settings[policy][2]
end

function get_policies(parsed::Dict)
    return get_policies(parsed["policy"])
end

function get_experience(buffer_size, μ, π; rng=Random.GLOBAL_RNG, chain_size=10, state_mod=identity)

    env = MarkovChain(chain_size)
    _, prev_state = start!(env; rng=rng)
    ϕ = state_mod(prev_state)

    er_names = [:s_t, :s_tp1, :a_t, :r, :γ_tp1, :terminal, :mu_t, :pi_t, :ρ]
    er_types = [typeof(ϕ), typeof(ϕ), Int64, Float64, Float64, Bool, Float64, Float64, Float64]
    ER = ExperienceReplay(buffer_size, er_types; column_names=er_names)
    WER = WeightedExperienceReplay(buffer_size, er_types; column_names=er_names)

    for i = 1:buffer_size

        action = StatsBase.sample(rng, get_actions(env), StatsBase.Weights(μ))
        _, state, reward, terminal = step!(env, action)

        experience = (state_mod(prev_state), state_mod(state), action, reward, 0.9, terminal,
                      μ[action], π[action], π[action]/μ[action])
        add!(ER, experience)
        add!(WER, experience, π[action]/μ[action])

        prev_state = copy(state)
        if terminal
            _, prev_state = start!(env; rng=rng)
        end
    end

    return ER, WER
end


function get_experience_interpolation(buffer_size, μ, π, sample_policy;
                                      rng=Random.GLOBAL_RNG, chain_size=10,
                                      state_mod=identity)

    env = MarkovChain(chain_size)
    _, prev_state = start!(env; rng=rng)
    ϕ = state_mod(prev_state)

    er_names = [:s_t, :s_tp1, :a_t, :r, :γ_tp1, :terminal, :mu_t, :pi_t, :ρ]
    er_types = [typeof(ϕ), typeof(ϕ), Int64, Float64, Float64, Bool, Float64, Float64, Float64]
    WER = WeightedExperienceReplay(buffer_size, er_types; column_names=er_names)

    for i = 1:buffer_size

        action = StatsBase.sample(rng, get_actions(env), StatsBase.Weights(μ))
        _, state, reward, terminal = step!(env, action)

        experience = (state_mod(prev_state), state_mod(state), action, reward, 0.9, terminal,
                      μ[action], π[action], π[action]/sample_policy[action])
        add!(WER, experience, sample_policy[action]/μ[action])

        prev_state = copy(state)
        if terminal
            _, prev_state = start!(env; rng=rng)
        end
    end

    return WER
end

end
