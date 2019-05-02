
using Pkg
Pkg.activate(".")

using Reproduce
using Logging

const save_loc = "four_rooms_cont_exp_normis"
const exp_file = "experiment/four_rooms_cont.jl"
const exp_module_name = :FourRoomsContExperiment
const exp_func_name = :main_experiment
const alphas = collect(0.0:0.05:1.0)
const policies = ["random_state_variant", "random_state_weight_variant", "uniform"]
const gvfs = ["collide_down", "favored_down"]
const batchsizes = [1, 8, 16]
const train_gaps = [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 48, 64]
const warm_up = 1000
const buffersize = 15000
const numsteps = 250000

function make_arguments(args::Dict{String, String})
    new_args=["--policy", args["policy"],
              "--gvf", args["gvf"],
              "--train_gap", args["train_gap"],
              "--batchsize", args["batchsize"],
              "--run", args["run"]]
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
        default=10
        "--saveloc"
        arg_type=String
        default=save_loc
        "--numjobs"
        action=:store_true
    end
    parsed = parse_args(as)
    num_workers = parsed["numworkers"]

    arg_dict = Dict([
        "policy"=>policies,
        "gvf"=>gvfs,
        "train_gap"=>train_gaps,
        "batchsize"=>batchsizes,
        "run"=>1:parsed["numruns"]
    ])
    arg_list = ["policy", "gvf", "train_gap", "batchsize", "run"]
    alg_list = ["--normis", "--is", "--ir",
                "--bcir",
                "--wisbuffer", "--wisbatch",
                "--vtrace", "--clip_value_perc", "0.5", "0.9", "1.0", "--clip_value", "1.0"]
    static_args = [alg_list;
                   ["--exp_loc", parsed["saveloc"],
                    "--warm_up", string(warm_up),
                    "--buffersize", string(buffersize),
                    "--seed", "0",
                    "--numinter", string(numsteps),
                    # "--noise_params", "0.1", "0.001",
                    "--eval_points", "100",
                    "--eval_steps", "100",
                    # "--norm_is",
                    "--compress",
                    "--alphas"]; string.(alphas./16)]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments)

    if parsed["numjobs"]
        @info "This experiment has $(length(collect(args_iterator))) jobs."
        println(collect(args_iterator)[num_workers])
        exit(0)
    end

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






