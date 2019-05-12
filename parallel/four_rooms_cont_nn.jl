#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH -o four_rooms_nn.out # Standard output
#SBATCH -e four_rooms_nn.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=6:00:00 # Running time of 6 hours
#SBATCH --ntasks=32
#SBATCH --account=def-whitem

using Pkg
Pkg.activate(".")

using Reproduce
using Logging

const save_loc = "four_rooms_cont_nn_exp"
const exp_file = "experiment/four_rooms_cont_nn.jl"
const exp_module_name = :FourRoomsContNNExperiment
const exp_func_name = :main_experiment
const alphas = [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
const policies = ["random_state_variant", "random_state_weight_variant", "uniform"]
const gvfs = ["collide_down", "favored_down"]
const batchsizes = [16]
const train_gaps = [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 48, 64, 128, 256]
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
    alg_list = ["--normis", "--is", "--ir", "--bcir", "--wsnormis", "--incnormis", "--wsavgnormis",
                "--wisbatch", "--vtrace", "--clip_value", "1.0"]
    static_args = [alg_list;
                   ["--exp_loc", parsed["saveloc"],
                    "--warm_up", string(warm_up),
                    "--seed", "0",
                    "--buffersize", string(buffersize),
                    "--numinter", string(numsteps),
                    "--eval_points", "100",
                    "--eval_steps", "100",
                    "--opt", "RMSProp",
                    "--compress"]]
                    # "--alphas"]; string.(collect(alphas))]
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






