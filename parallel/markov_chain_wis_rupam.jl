
using Pkg
Pkg.activate(".")

using Reproduce
using Logging

const save_loc = "markov_chain_results"
const exp_file = "experiment/markov_chain.jl"
const exp_module_name = :MarkovChainExperiment
const exp_func_name = :main_experiment
# const alphas = collect(0.0:0.1:12)
const u_0 = 10 .^ (0:0.25:3)
const alphas = [0.0, 0.001, 0.1, 1.0]
const policies = ["easy", "hard", "hardest"]

function make_arguments(args::Dict)
    new_args=["--policy", args["policy"]]
    return new_args
end

function main()

    as = ArgParseSettings()
    @add_arg_table as begin
        "numworkers"
        arg_type=Int64
        default=1
        "--numruns"
        arg_type=Int64
        default=100
        "--saveloc"
        arg_type=String
        default=save_loc
        "--numjobs"
        action=:store_true
    end
    parsed = parse_args(as)
    num_workers = parsed["numworkers"]
    
    arg_dict = Dict([
        # "outhorde"=>["onestep", "chain"],
        "policy"=>policies
    ])
    arg_list = ["policy"]
    alg_list = ["--wisrupam", "--normalize_eta"]
    static_args = [alg_list; ["--exp_loc", parsed["saveloc"], "--numruns", string(parsed["numruns"]), "--numiter", "1000", "--alphas", string.(alphas)..., "--init_u", string.(u_0)...]]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments)

    if parsed["numjobs"]
        @info "This experiment has $(length(collect(args_iterator))) jobs."
        println(collect(args_iterator)[num_workers])
        exit(0)
    end

    # experiment = Experiment(save_loc)
    experiment = Experiment(save_loc,
                            exp_file,
                            exp_module_name,
                            exp_func_name,
                            args_iterator)

    create_experiment_dir(experiment)
    add_experiment(experiment; settings_dir="settings")
    ret = job(experiment; num_workers=num_workers)
    post_experiment(experiment, ret)
end


main()
