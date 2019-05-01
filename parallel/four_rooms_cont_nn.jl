
using Pkg
Pkg.activate(".")

using Reproduce
using Logging

const save_loc = "four_rooms_cont_nn_exp"
const exp_file = "experiment/four_rooms_cont_nn.jl"
const exp_module_name = :FourRoomsContNNExperiment
const exp_func_name = :main_experiment
const alphas = collect(0.0:0.1:2.0)
const policies = ["random_state_variant", "random_state_weight_variant", "uniform"]
const gvfs = ["collide_down", "favored_down"]
const batchsizes = [1, 8, 16]
const train_gaps = [1, 8, 16]
const warm_up = 1000
const buffersize = 15000
const numsteps = 250000

function make_arguments(args::Dict{String, String})
    new_args=["--policy", args["policy"],
              "--gvf", args["gvf"],
              "--train_gap", args["train_gap"],
              "--batchsize", args["batchsize"],
              "--optparams", args["alpha"],
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
        "alpha"=>alphas,
        "run"=>1:parsed["numruns"]
    ])
    arg_list = ["policy", "gvf", "train_gap", "batchsize", "alpha", "run"]
    alg_list = ["--is", "--ir", "--bcir"]
    static_args = [alg_list;
                   ["--exp_loc", parsed["saveloc"],
                    "--warm_up", string(warm_up),
                    "--seed", "0",
                    "--buffersize", string(buffersize),
                    "--numinter", string(numsteps),
                    "--"
                    "--eval_points", "100",
                    "--eval_steps", "100",
                    "--opt", "Descent",
                    "--compress"]]
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






