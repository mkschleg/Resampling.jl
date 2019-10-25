
using JLD2
using Plots
using ProgressMeter

include("experiments/markov_chain_var.jl")
include("plot/plot_utils.jl")

function main()

    plot_settings = Dict{Symbol, Any}(
        :linewidth=>2,
        # :color=>[:purple, :blue, :purple, :blue, :green],
        # :linestyle=>[:solid, :solid, :dot, :dot, :solid],
        :color=>[:green, :blue, :green, :purple],
        :linestyle=>[:solid, :solid, :dot, :solid],
        :grid=>false,
        :size=>(432,244),
        :tick_direction=>:out,
        # :ylim=>(0,0.2),
        :legend=>false
    )
    policies = ["easy", "hard", "hardest"]
    # policies = ["hardest"]
    num_runs = 100
    measure_steps=0:5:250
    x = convert(Array{Float64,1}, collect(measure_steps))
    lck = Threads.Mutex()

    # plot_algorithms = ["IS", "IR", "BC-IR", "WIS-Buffer", "WIS-Batch"]
    plot_algorithms = ["IS", "IR", "VTrace_1.0", "WIS-Batch"]

    # Threads.@threads for policy in policies
    @showprogress 0.1 "Policy: " for policy in policies
    # for policy in policies
        ret = measure_variance_over_time_norm(
            num_runs, 15000, 1000, 16,
            policy_settings[policy][1], policy_settings[policy][2];
            measure_steps=measure_steps, random_value=false, norm_l=1)
        lock(lck)

        p = plot_dict(x,
                      filter(p -> p[1] in plot_algorithms, ret[2]),
                      filter(p -> p[1] in plot_algorithms, ret[3]);
                      plot_order=plot_algorithms,
                      plot_settings...)
        savefig(p, "markov_chain_variance_l1_$(policy).pdf")
        @save "markov_chain_variance_l1_$policy.jld" ret
        unlock(lck)
    end


end

main()
