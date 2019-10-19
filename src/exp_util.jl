
glorot_uniform(rng::Random.AbstractRNG, dims...) = begin; (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims)) end;
glorot_normal(rng::Random.AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))


module ExpUtils

import ..JuliaRL
using ..Resampling
using Reproduce


export algorithm_settings!, build_algorithm_dict


function algorithm_settings!(s::ArgParseSettings)
    @add_arg_table s begin
        # Algorithms
        "--is"
        action = :store_true
        help = "set to include base off-policy importance sampling"
        "--normis"
        action = :store_true
        help = "set to include base off-policy importance sampling"
        "--incnormis"
        action = :store_true
        help = "set to include base off-policy importance sampling"
        "--wsnormis"
        action = :store_true
        help = "set to include base off-policy importance sampling"
        "--wsavgnormis"
        action = :store_true
        help = "set to include base off-policy importance sampling"
        "--ir"
        action = :store_true
        help = "Set to include ir"
        "--bcir"
        action = :store_true
        help = "Set to include buffer corrected ir"
        "--wisbatch"
        action = :store_true
        help = "Set to include wisbatch in experiments"
        "--wisbuffer"
        action = :store_true
        help = "Set to include wisbuffer in experiments"
        "--wisoptimal"
        action = :store_true
        help = "Set to run wis optimal experiments"
        "--wisrupam"
        action = :store_true
        help = "Set to run wis rupam experiments"
        "--sarsa"
        action = :store_true
        help = "Set to run sarsa"
        "--expectedsarsa"
        action = :store_true
        help = "Set to run expected sarsa"
        "--vtrace"
        action = :store_true
        help = "Set to run clipped is"
        "--clip_value"
        nargs='+'
        arg_type = Float64
        "--clip_value_perc"
        nargs='+'
        arg_type = Float64
    end
    return s
end

function build_algorithm_dict(parsed; max_is=1.0, size_features = 0)
    algo_dict = Dict{String, Resampling.LearningUpdate}()
    sample_dict = Dict{String, String}()
    value_type_dict = Dict{String, String}()

    if parsed["is"]
        algo_dict["IS"] = BatchTD()
        sample_dict["IS"] = "ER"
        value_type_dict["IS"] = "State"
    end
    if parsed["normis"]
        algo_dict["NormIS"] = BatchTD()
        sample_dict["NormIS"] = "ER"
        value_type_dict["NormIS"] = "State"
    end
    if parsed["incnormis"]
        algo_dict["IncNormIS"] = IncNormIS()
        sample_dict["IncNormIS"] = "ER"
        value_type_dict["IncNormIS"] = "State"
    end
    if parsed["wsnormis"]
        algo_dict["WSNormIS"] = WSNormIS()
        sample_dict["WSNormIS"] = "ER"
        value_type_dict["WSNormIS"] = "State"
    end
    if parsed["wsavgnormis"]
        algo_dict["WSAvgNormIS"] = WSNormIS()
        sample_dict["WSAvgNormIS"] = "ER"
        value_type_dict["WSAvgNormIS"] = "State"
    end
    if parsed["ir"]
        algo_dict["IR"] = BatchTD()
        sample_dict["IR"] = "IR"
        value_type_dict["IR"] = "State"
    end
    if parsed["bcir"]
        algo_dict["BCIR"] = BatchTD()
        sample_dict["BCIR"] = "IR"
        value_type_dict["BCIR"] = "State"
    end
    if parsed["wisbatch"]
        algo_dict["WISBatch"] = WISBatchTD()
        sample_dict["WISBatch"] = "ER"
        value_type_dict["WISBatch"] = "State"
    end
    if parsed["wisbuffer"]
        algo_dict["WISBuffer"] = BatchTD()
        sample_dict["WISBuffer"] = "ER"
        value_type_dict["WISBuffer"] = "State"
    end

    if parsed["wisoptimal"]
        algo_dict["WISOptimal"] = WISBatchTD()
        sample_dict["WISOptimal"] = "Optimal"
        value_type_dict["WISOptimal"] = "State"
    end

    if parsed["wisrupam"]
        for u_0 in parsed["init_u"]
            algo_dict["WISRupam"] = Resampling.WISBatchTD_Rupam()
            sample_dict["WISRupam"] = "ER"
            value_type_dict["WISRupam"] = "State"
        end
    end

    if parsed["vtrace"]
        for clip_value in parsed["clip_value"]
            algo_dict["VTrace_$(clip_value)"] = VTrace(clip_value)
            sample_dict["VTrace_$(clip_value)"] = "ER"
            value_type_dict["VTrace_$(clip_value)"] = "State"
        end
        for clip_value_perc in parsed["clip_value_perc"]
            algo_dict["VTrace_$(clip_value_perc*max_is)"] = VTrace(clip_value_perc*max_is)
            sample_dict["VTrace_$(clip_value_perc*max_is)"] = "ER"
            value_type_dict["VTrace_$(clip_value_perc*max_is)"] = "State"
        end
    end

    if parsed["sarsa"]
        algo_dict["Sarsa"] = BatchSarsa()
        sample_dict["Sarsa"] = "ER"
        value_type_dict["Sarsa"] = "StateAction"
    end

    if parsed["expectedsarsa"]
        algo_dict["ExpectedSarsa"] = BatchExpectedSarsa()
        sample_dict["ExpectedSarsa"] = "ER"
        value_type_dict["ExpectedSarsa"] = "StateAction"
    end
    return algo_dict, sample_dict, value_type_dict
end

export MarkovChainUtil
include("exp_utils/MarkovChain.jl")

export FourRoomsUtil
include("exp_utils/FourRooms.jl")

export FourRoomsContUtil
include("exp_utils/FourRoomsCont.jl")

export FluxUtil
include("exp_utils/Flux.jl")


end
