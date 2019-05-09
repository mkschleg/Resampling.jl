
__precompile__(true)

module MarkovChainIntExperiment

using Reproduce
using JLD2
using Resampling
import Resampling.ExpUtils: algorithm_settings!, build_algorithm_dict
using Resampling.ExpUtils.MarkovChainUtil
using Random
using Statistics
import Flux: Descent
import ProgressMeter
import StatsBase


function exp_settings(as::ArgParseSettings = ArgParseSettings(exc_handler=Reproduce.ArgParse.debug_handler))
    @add_arg_table as begin
        # Basic Experiment Arguments
        "--exp_loc"
        arg_type=String
        required=true
        "--run"
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
        "--onpolicy"
        action=:store_true
        "--working"
        action=:store_true
    end

    @add_arg_table as begin
        "--alphas"
        arg_type=Float64
        nargs='+'
        required=true
    end

    @add_arg_table as begin
        "--beta"
        help="Interpolation between policies"
        arg_type=Float64
        required=true
    end

    # algorithm_settings!(as)
    env_settings!(as)

    return as
end

function rmse(model::Resampling.TabularLayer{Array{Float64, 1}}, truth, args...)
    return mean(abs.(model.W .- truth))
end

function rmse(model::Resampling.TabularLayer{Array{Float64, 2}}, truth, target_policy)
    return mean(abs.(model.W'*target_policy  .- truth))
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

    run = parsed["run"]
    chain_size = parsed["chainsize"]
    num_iterations = parsed["numiter"]
    buffer_size = parsed["buffersize"]
    batch_size = parsed["batchsize"]
    α_arr = parsed["alphas"]
    β = parsed["beta"]

    μ, π = get_policies(parsed)
    if parsed["onpolicy"]
        μ = π
    end

    truth = Resampling.DynamicProgramming(MarkovChain(chain_size), π, 0.9)

    lu = BatchTD()
    value_arr = [Tabular(chain_size) for α in α_arr]
    local_error = zeros(length(α_arr), num_iterations)

    rng = MersenneTwister(parsed["seed"] + run)
    WER = get_experience_interpolation(buffer_size, μ, π, β*μ + (1.0-β)*π; rng=rng, chain_size=chain_size)

    for iter in 1:num_iterations

        samp_wer = sample(WER, batch_size; rng=rng)
        arg_wer = (samp_wer[:ρ], samp_wer[:s_t], samp_wer[:s_tp1],
                   samp_wer[:r], samp_wer[:γ_tp1], samp_wer[:terminal])

        for (α_idx, α) in enumerate(α_arr)
            opt = Descent(α)
            update!(value_arr[α_idx], opt, lu, arg_wer...;)
            local_error[α_idx, iter] = rmse(value_arr[α_idx], truth, π)
        end
    end


    # println(value_function)

    if parsed["working"]
        return local_error
    else
        results = local_error
        @save joinpath(save_loc, "results.jld2") results
    end

end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end







