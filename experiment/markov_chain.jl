
__precompile__(true)

module MarkovChainExperiment

using Reproduce
using JLD2
using Resampling
import Resampling.ExpUtils: algorithm_settings!, build_algorithm_dict
using Resampling.ExpUtils.MarkovChainUtil
using Random
using Statistics
import Flux: Descent
import ProgressMeter


function exp_settings(as::ArgParseSettings = ArgParseSettings(exc_handler=Reproduce.ArgParse.debug_handler))
    @add_arg_table as begin
        # Basic Experiment Arguments
        "--exp_loc"
        arg_type=String
        required=true
        "--numruns"
        arg_type=Int64
        required=true
        "--seed"
        help = "Starting seed (initialize seed + run)"
        default = 0
        arg_type = Int64
        "--numiter"
        help = "Number of training iterations"
        default = 1000
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

function rmse(model::Resampling.TabularLayer, truth)
    return mean(abs.(model.W .- truth))
end

function main_experiment(args::Vector{String})

    as = exp_settings()
    parsed = parse_args(args, as)
    if !parsed["working"]
        create_info!(parsed, parsed["exp_loc"]; filter_keys=["verbose", "working", "exp_loc"])
        save_loc = Reproduce.get_save_dir(parsed)
        if isfile(joinpath(save_loc, "results.jld2"))
            return
        end
    end

    num_runs = parsed["numruns"]
    chain_size = parsed["chainsize"]
    num_iterations = parsed["numiter"]
    buffer_size = parsed["buffersize"]
    batch_size = parsed["batchsize"]
    α_arr = parsed["alphas"]

    μ, π = get_policies(parsed)
    truth = Resampling.DynamicProgramming(MarkovChain(chain_size), π, 0.9)
    println(truth)

    algo_dict, sample_dict, value_type_dict = build_algorithm_dict(parsed; max_is=max_is_dict[parsed["policy"]])

    value_dict = Dict{String, Array{Resampling.TabularLayer, 1}}()
    error_dict = Dict{String, Array{Float64}}()
    for key in keys(algo_dict)
        if value_type_dict[key] == "State"
            value_dict[key] = [Tabular(chain_size) for α in α_arr]
        else
            value_dict[key] = [Tabular(2, chain_size) for α in α_arr]
        end
        error_dict[key] = zeros(num_runs, length(α_arr), num_iterations)
    end

    ProgressMeter.@showprogress 0.1 "run: " for run in 1:num_runs
        rng = MersenneTwister(parsed["seed"] + run)
        for vf in values(value_dict)
            for v in vf
                fill!(v.W, 0.0)
            end
        end
        ER, WER = get_experience(buffer_size, μ, π; rng=rng, chain_size=chain_size)
        avg_is = Resampling.total(WER.sumtree)/buffer_size

        for iter in 1:num_iterations
            samp_er = sample(ER, batch_size; rng=rng)
            samp_wer = sample(WER, batch_size; rng=rng)

            arg_er = (samp_er[:ρ], samp_er[:s_t], samp_er[:s_tp1], samp_er[:r], samp_er[:γ_tp1], samp_er[:terminal], samp_er[:a_t], π)
            arg_wer = (ones(length(samp_wer[:ρ])), samp_wer[:s_t], samp_wer[:s_tp1], samp_wer[:r], samp_wer[:γ_tp1], samp_wer[:terminal], samp_er[:a_t], π)

            for (α_idx, α) in enumerate(α_arr)
                opt = Descent(α)
                for key in keys(algo_dict)
                    if sample_dict[key] == "IR"
                        corr_term = 1.0
                        if key == "BCIR"
                            corr_term = avg_is
                        end
                        update!(value_dict[key][α_idx], opt, algo_dict[key], arg_wer...; corr_term=corr_term)
                    else sample_dict[key] == "ER"
                        update!(value_dict[key][α_idx], opt, algo_dict[key], arg_er...;)
                    end
                    error_dict[key][run, α_idx, iter] = rmse(value_dict[key][α_idx], truth)
                end
            end
        end
    end

    # println(value_function)

    results = Dict{String, Dict{String, Float64}}()
    results["mode"] = Dict{String, Float64}()
    results["min"] = Dict{String, Float64}()
    results["max"] = Dict{String, Float64}()
    results["mean"] = Dict{String, Float64}()
    results["std"] = Dict{String, Float64}()

    if parsed["working"]
        return error_dict
    else
        # Save or something...
    end

end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end







