
__precompile__(true)

module FourRoomsExperiment

using Reproduce
using JLD2
using Resampling
import Resampling.ExpUtils: algorithm_settings!, build_algorithm_dict
using Resampling.ExpUtils.FourRoomsUtil
# import JuliaRL
using Random
using Statistics
import Flux: Descent
import ProgressMeter
import StatsBase

function mave(model::Resampling.TabularLayer{Array{Float64, 2}}, truth, args...)
    return mean(abs.(model.W .- truth))
end

function mave(model::Resampling.TabularLayer{Array{Float64, 3}}, truth, target_policy_matrix)
    # println(size(model.W))
    return @inbounds mean(abs.(sum(target_policy_matrix .* model.W; dims=1)[1,:,:]  .- truth))
end

function get_target_policy_matrix(env::FourRooms, μ::Resampling.AbstractPolicy)
    tmp = zeros(4, 11, 11)
    for state in Resampling.get_states(env)
        tmp[:, state] .= [get(μ, state, action, nothing, nothing, nothing) for action in 1:4]
    end
    return tmp
end


function exp_settings(as::ArgParseSettings = ArgParseSettings(exc_handler=Reproduce.ArgParse.debug_handler))
    @add_arg_table as begin
        # Basic Experiment Arguments
        "--exp_loc"
        arg_type=String
        required=true
        "--seed"
        help = "Starting seed (initialize seed + run)"
        default = 0
        arg_type = Int64
        "--run"
        help = "Run"
        default = 1
        arg_type = Int64
        "--numinter"
        help = "Number of interactions"
        default = 1000
        arg_type = Int64
        "--train_gap"
        help = "Number of interactions in-between training iterations"
        default = 1
        arg_type = Int64
        "--warm_up"
        help = "number of interactions to wait to begin learning"
        default = 0
        arg_type = Int64
        "--buffersize"
        help = "Size of buffer"
        default = 15000
        arg_type = Int64
        "--batchsize"
        help = "Size of batch"
        default = 16
        arg_type = Int64
        "--working"
        action=:store_true
        "--compress"
        action=:store_true
    end

    @add_arg_table as begin
        "--alphas"
        arg_type=Float64
        nargs='+'
        required=true
    end

    algorithm_settings!(as)
    env_settings!(as)

    return as
end


function main_experiment(args::Vector{String})

    as = exp_settings()
    parsed = parse_args(args, as)
    save_loc=""
    if !parsed["working"]
        create_info!(parsed, parsed["exp_loc"]; filter_keys=["verbose", "working", "exp_loc"])
        save_loc = Reproduce.get_save_dir(parsed)
        if isfile(joinpath(save_loc, "results.jld2"))
            return
        end
    end

    # num_runs = parsed["numruns"]
    train_gap = parsed["train_gap"]
    warm_up = parsed["warm_up"]
    num_interactions = parsed["numinter"]
    buffer_size = parsed["buffersize"]
    batch_size = parsed["batchsize"]
    α_arr = parsed["alphas"].*batch_size

    μ = get_policy(parsed)
    gvf = FourRoomsUtil.GVFS[parsed["gvf"]]

    # get policy matrix over all states:
    env = FourRooms()
    target_policy_matrix = get_target_policy_matrix(env, gvf.policy)

    truth = Resampling.DynamicProgramming(env, gvf)
    agent = FourRoomsAgent(μ, gvf, α_arr, train_gap, buffer_size, batch_size, warm_up, parsed, size(env))

    error_dict = Dict{String, Array{Float64}}()
    for key in keys(agent.algo_dict)
        error_dict[key] = zeros(Float32, length(α_arr), num_interactions)
    end

    rng = MersenneTwister(parsed["seed"] + parsed["run"])

    _, s_t = start!(env; rng=rng)
    action = start!(agent, s_t; rng=rng)

    # ProgressMeter.@showprogress 0.1 "Step: " for step in 1:num_interactions
    for step in 1:num_interactions

        # Get experience from environment.
        _, s_tp1, r, terminal = step!(env, action; rng=rng)
        # println(s_tp1, action)
        action = step!(agent, s_tp1, r, terminal; rng=rng)

        # calculate error
        for key in keys(agent.algo_dict)
            for α_idx in eachindex(agent.α_arr)
                error_dict[key][α_idx, step] = Float32(mave(agent.value_dict[key][α_idx], truth, target_policy_matrix))
            end
        end
    end
    # println(value_function)

    if parsed["working"]
        return error_dict
    else
        results = error_dict
        if parsed["compress"]
            JLD2.save(
                JLD2.FileIO.File(JLD2.FileIO.DataFormat{:JLD2},
                                 joinpath(save_loc, "results.jld2")),
                Dict("results"=>results); compress=true)
        else
            @save joinpath(save_loc, "results.jld2") results
        end
    end

end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end







