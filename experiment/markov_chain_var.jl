
using ProgressMeter
using JLD2
using ArgParse
using Random
using Statistics
using LinearAlgebra

using Resampling
using Resampling: TabularRL
using Resampling.Environments

import StatsBase

function measure_variance_sample_martha_time_norm(iterations, batch_size, ER, WER;
                                                  rng=Random.GLOBAL_RNG,
                                                  chain_size=10,
                                                  random_value=false,
                                                  value_function=nothing,
                                                  norm_l=2)

    sum_is = IWER.total(WER.sumtree)
    avg_is = sum_is/size(WER)[1]
    if value_function == nothing
        value_function = TabularRL.StateValueFunction(chain_size)
        if random_value
            value_function.values = randn(rng, chain_size)
        else
            fill!(value_function.values, 0.5)
        end
    end

    update_error = Dict{String, Array{Float64}}()

    update_error["IS"] = zeros(iterations)
    update_error["IR"] = zeros(iterations)
    update_error["VTrace_1.0"] = zeros(iterations)
    update_error["BC-IR"] = zeros(iterations)
    update_error["WIS-Batch"] = zeros(iterations)
    update_error["WIS-Buffer"] = zeros(iterations)
    update_error["WIS-Buffersum"] = zeros(iterations)


    # sample
    for i in 1:iterations
        s_er = sample(ER, batch_size; rng=rng)
        s_wer = sample(WER, batch_size; rng=rng)
        batch_is_sum = sum(s_er[:,9])

        # Note: Experience order
        # (state_t[1], state_tp1[1], action, reward, gamma, terminal,
        #   mu[action + 1], pi[action + 1], pi[action + 1]/mu[action + 1])

        args_er = (s_er[:, 9], s_er[:,1], s_er[:,2], s_er[:,4], s_er[:,5], s_er[:,6])
        args_er = (s_er[:, 9], s_er[:,1], s_er[:,2], s_er[:,4], s_er[:,5], s_er[:,6])
        args_wer = (ones(batch_size), s_wer[:,1], s_wer[:,2], s_wer[:,4], s_wer[:,5], s_wer[:,6])

        update_error["IS"][i] = norm(mean(tdupdate(value_function, args_er...)), norm_l)
        update_error["VTrace_1.0"][i] = norm(mean(tdupdate_vtrace(1.0, value_function, args_er...)), norm_l)
        update_error["IR"][i] = norm(mean(tdupdate(value_function, args_wer...)), norm_l)
        update_error["BC-IR"][i] = norm(mean((1.0/avg_is) .* tdupdate(value_function, args_wer...)), norm_l)
        update_error["WIS-Buffersum"][i] = norm(sum((1.0/sum_is) .* tdupdate(value_function, args_er...)), norm_l)
        update_error["WIS-Buffer"][i] = norm(mean((size(ER)[1]/sum_is) .* tdupdate(value_function, args_er...)), norm_l)
        update_error["WIS-Batch"][i] = norm(sum((1.0/(sum(s_er[:,end]))) .* tdupdate(value_function, args_er...)), norm_l)
    end

    update_variance = Dict{String, Float64}()

    for key in keys(update_error)
        update_variance[key] = var(update_error[key])
    end

    return update_variance

end

function add_res_to_dict!(num_runs, r, res_dict, res)
    if length(keys(res_dict)) == 0
            # res_dict = typeof(res)()
        for k in keys(res)
            res_dict[k] = zeros(num_runs)
            res_dict[k][r] = res[k]
        end
    else
        for k in keys(res)
            res_dict[k][r] = res[k]
        end
    end
end

function create_ret_dict(num_runs, res_dict)

    ret_dict = Dict{String, Array{Float64,1}}()
    for k in keys(res_dict)
        ret_dict[k] = [mean(res_dict[k]), std(res_dict[k])/sqrt(num_runs)]
    end
    return ret_dict
end


function measure_variance_over_time_norm(num_runs, buffer_size, iterations, batch_size, mu, pi; measure_steps=[1000], gamma=0.9, chain_size=10, verbose=false, kwargs...)

    func = measure_variance_sample_martha_time_norm
    truth = MarkovChain.DynamicProgramming(MarkovChain.environment(10), pi, gamma)
    res_dict = Dict{Int64, Dict{String, Array{Float64, 1}}}()
    for step in measure_steps
        res_dict[step] = Dict{String, Array{Float64, 1}}()
    end

    @showprogress 0.1 "Runs: " for r in 1:num_runs
    # for r in 1:num_runs
        rng=Random.MersenneTwister(r)
        ER, WER = get_markov_chain_experience(buffer_size, mu, pi; rng=rng)
        sum_is = IWER.total(WER.sumtree)
        algo = BatchTD(1.0, buffer_size, 1.0/sum_is)
        value_function = TabularRL.StateValueFunction(chain_size)
        fill!(value_function.values, 0.0)

        for iter in 0:maximum(measure_steps)
            if iter in measure_steps
                res = func(iterations, batch_size, ER, WER;
                           rng=rng, value_function=value_function,
                           chain_size=chain_size, kwargs...)
                add_res_to_dict!(num_runs, r, res_dict[iter], res)
            end
            s_opt = IWER.getrow(ER.buffer, :)
            TabularRL.update!(value_function, algo,
                              s_opt[:, 9], s_opt[:, 1],
                              s_opt[:, 2], s_opt[:, 4],
                              s_opt[:, 5], s_opt[:, 6])
            if verbose
                println(iter, " ", rmse(value_function, pi, truth)[1])
            end
        end
    end

    ret_dict = Dict{Int64, Any}()
    for step in measure_steps
        ret_dict[step] = create_ret_dict(num_runs, res_dict[step])
    end

    time_series_dict = Dict{String, Array{Float64, 1}}()
    time_series_stderr_dict = Dict{String, Array{Float64, 1}}()
    for k in keys(ret_dict[measure_steps[1]])
        time_series_dict[k] = [ret_dict[s][k][1] for s in measure_steps]
        time_series_stderr_dict[k] = [ret_dict[s][k][2] for s in measure_steps]
    end


    return ret_dict, time_series_dict, time_series_stderr_dict
end



