
using ..JuliaRL.FeatureCreators
import ..build_algorithm_dict
import Flux: Descent

"""
    Agent for learning online in the four rooms environment.
"""
mutable struct TCFourRoomsContAgent{O, P<:Resampling.AbstractPolicy} <: JuliaRL.AbstractAgent
    μ::P
    ER::ExperienceReplay
    WER::WeightedExperienceReplay
    opt::O
    tilecoder::TileCoder
    gvf::GVF
    training_gap::Int64
    buffer_size::Int64
    batch_size::Int64
    warm_up::Int64
    algo_dict::Dict{String, Resampling.LearningUpdate}
    sample_dict::Dict{String, String}
    value_dict::Dict{String, Array{Resampling.SparseLayer, 1}}
    counter::Int64
    state_t::Tuple{Array{Float64, 1}, Bool}
    action_t::Int64
    α_arr::Array{Float64, 1}
    max_is::Float64
    function TCFourRoomsContAgent(μ::P, gvf::GVF, opt::O, num_tilings, num_tiles, α_arr::Array{Float64, 1},
                                  training_gap, buffer_size, batch_size,
                                  warm_up, parsed, env_size; max_is_ratio=1.0) where {O, P<:Resampling.AbstractPolicy}

        # build tile coder....
        tilecoder = TileCoder(num_tilings, num_tiles, 2)
        num_features = num_tilings*(num_tiles+1)^2
        algo_dict, sample_dict, value_type_dict = build_algorithm_dict(parsed; max_is=max_is_ratio, size_features=num_features)
        # println(algo_dict)
        value_dict = Dict{String, Array{Resampling.SparseLayer, 1}}()
        for key in keys(algo_dict)
            if value_type_dict[key] == "State"
                value_dict[key] = [SparseLayer(num_features, 1) for α in α_arr]
            else
                value_dict[key] = [SparseLayer(num_features, 4) for α in α_arr]
            end
        end
        er_names = [:s_t, :s_tp1, :a_t, :a_tp1, :r, :γ_tp1, :terminal, :mu_t, :pi_t, :ρ]
        state_type = Array{Int64, 1}
        er_types = [state_type, state_type, Int64, Int64, Float64, Float64, Bool, Float64, Float64, Float64]
        ER = ExperienceReplay(buffer_size, er_types; column_names=er_names)
        WER = WeightedExperienceReplay(buffer_size, er_types; column_names=er_names)
        new{O, P}(μ, ER, WER, opt, tilecoder, gvf, training_gap, buffer_size, batch_size, warm_up,
                  algo_dict, sample_dict, value_dict, warm_up, ([0.0,0.0], false), 0, α_arr,
                  max_is_ratio)
    end
end

JuliaRL.get_action(agent::TCFourRoomsContAgent, s; rng=Random.GLOBAL_RNG) =
    StatsBase.sample(rng, agent.μ, s[1])

function JuliaRL.start!(agent::TCFourRoomsContAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)
    agent.counter = agent.warm_up
    agent.state_t = env_s_tp1
    agent.action_t = JuliaRL.get_action(agent, env_s_tp1; rng=rng)
    return agent.action_t
end

function JuliaRL.step!(agent::TCFourRoomsContAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)

    # After taking action at timestep t
    μ_prob = get(agent.μ, agent.state_t[1], agent.action_t, nothing, nothing, nothing)
    c, γ, π_prob = get(agent.gvf, agent.state_t, agent.action_t, env_s_tp1, nothing, nothing)
    action_tp1 = JuliaRL.get_action(agent, agent.state_t; rng=rng)

    experience = (create_features(agent.tilecoder, agent.state_t[1]./11.0),
                  create_features(agent.tilecoder, env_s_tp1[1]./11.0),
                  agent.action_t, action_tp1, c, γ, terminal, μ_prob, π_prob, (π_prob/μ_prob))

    add!(agent.ER, experience)
    add!(agent.WER, experience, π_prob/μ_prob)

    if agent.counter == 0
        agent.counter = agent.training_gap
        train_value_functions(agent; rng=rng)
    end
    agent.counter -= 1

    agent.state_t = env_s_tp1
    agent.action_t = action_tp1 #JuliaRL.get_action(agent, agent.state_t; rng=rng)
    return agent.action_t
end

function train_value_functions(agent::TCFourRoomsContAgent; rng=Random.GLOBAL_RNG)

    α_arr = agent.α_arr

    samp_er = sample(agent.ER, agent.batch_size; rng=rng)
    samp_wer = sample(agent.WER, agent.batch_size; rng=rng)

    arg_er = (samp_er[:ρ], samp_er[:s_t], samp_er[:s_tp1],
              samp_er[:r], samp_er[:γ_tp1], samp_er[:terminal],
              samp_er[:a_t], samp_er[:a_tp1], agent.gvf.policy)

    arg_wer = (ones(length(samp_wer[:ρ])), samp_wer[:s_t], samp_wer[:s_tp1],
               samp_wer[:r], samp_wer[:γ_tp1], samp_wer[:terminal],
               samp_wer[:a_t], samp_wer[:a_tp1], agent.gvf.policy)

    for (α_idx, α) in enumerate(α_arr)
        agent.opt.eta = α
        for key in keys(agent.algo_dict)
            if agent.sample_dict[key] == "IR"
                corr_term = 1.0
                if key == "BCIR"
                    corr_term = Resampling.total(agent.WER)/size(agent.WER)[1]
                end
                update!(agent.value_dict[key][α_idx], agent.opt, agent.algo_dict[key], arg_wer...; corr_term=corr_term)
            elseif agent.sample_dict[key] == "ER"
                if key == "WISBuffer"
                    corr_term = Resampling.total(agent.WER)/size(agent.WER)[1]
                    update!(agent.value_dict[key][α_idx], agent.opt, agent.algo_dict[key], arg_er...; corr_term=1.0/corr_term)
                elseif key == "NormIS"
                    update!(agent.value_dict[key][α_idx], agent.opt, agent.algo_dict[key], arg_er[1]./agent.max_is, arg_er[2:end]...;)
                else
                    update!(agent.value_dict[key][α_idx], agent.opt, agent.algo_dict[key], arg_er...;)
                end
            elseif agent.sample_dict[key] == "Optimal"
                samp_opt = agent.ER.buffer
                arg_opt = (samp_opt[:ρ], samp_opt[:s_t], samp_opt[:s_tp1],
                           samp_opt[:r], samp_opt[:γ_tp1], samp_opt[:terminal])
                update!(agent.value_dict[key][α_idx], agent.opt, agent.algo_dict[key], arg_opt...;)
            end
        end
    end
end

function predict(agent::TCFourRoomsContAgent, key, α_idx, eval_states::Array{Array{Float64, 1}, 1})
    states = [create_features(agent.tilecoder, eval_state./11.0) for eval_state in eval_states]
    return agent.value_dict[key][α_idx].(states)
end


function predict!(agent::TCFourRoomsContAgent, eval_states::Array{Array{Float64, 1}, 1}, predict_dict::Dict{String, Array{Array{Float64, 1}}})
    states = [create_features(agent.tilecoder, eval_state./11.0) for eval_state in eval_states]
    if length(keys(predict_dict)) == 0
        for key in keys(agent.value_dict)
            predict_dict[key] = [zeros(length(eval_states)) for i in 1:length(agent.α_arr)]
        end
    end
    for key in keys(agent.value_dict)
        for α_idx in 1:length(agent.α_arr)
            predict_dict[key][α_idx] .= agent.value_dict[key][α_idx].(states)
        end
    end
end





