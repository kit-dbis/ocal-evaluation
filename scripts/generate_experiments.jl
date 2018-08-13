
!isempty(ARGS) || error("No config supplied.")
isfile(ARGS[1]) || error("Cannot read '$(ARGS[1])'")
isabspath(ARGS[1]) || error("Please use an absolute path for the config.")
println("Config supplied: '$(ARGS[1])'")
config_file = ARGS[1]

using SVDD, OneClassActiveLearning
using Gurobi
using MLKernels
using JLD
using Memento

srand(0)
include(config_file)

function generate_experiments(data_files, data_split_strategy, initial_pool_strategy, num_resamples_initial_pool=5)
    exp_name = "$(data_split_strategy)-$(initial_pool_strategy)-$(data_info)"
    exp_dir = data_output_root * exp_name * "/"
    println(exp_dir)
    if isdir(exp_dir)
        print("Type 'yes' or 'y' to delete and overwrite experiment $(exp_dir): ")
        argin = readline()
        if argin == "yes" || argin == "y"
            rm(exp_dir, recursive=true)
        else
            error("Overwriting anyways... Just kidding, nothing happened.")
        end
    else
        !isdir(exp_dir) || error("The experiment directory $(exp_dir) already exists.")
    end

    isdir(exp_dir) || mkpath(exp_dir)
    mkpath("$(exp_dir)log/experiment/")
    mkpath("$(exp_dir)log/worker/")
    mkpath("$(exp_dir)results/")
    info("Generating experiment directory with name: $exp_dir and config: $(config_file). ($num_resamples_initial_pool resamples of the initial pool)")

    experiments = []

    for data_file in data_files
        data, labels = load_data(data_file)
        target_outlier_percentage = sum(labels .== :outlier) / length(labels)

        for n in 1:num_resamples_initial_pool
            srand(n)
            split_strategy, initial_pools = get_splits_and_init_pools(data, labels, data_split_strategy, initial_pool_strategy)
            param = Dict(:num_al_iterations => num_al_iterations,
                         :solver => GurobiSolver(OutputFlag=0, Threads=1),
                         :initial_pools => initial_pools,
                         :adjust_K => true,
                         :initial_pool_resample_version => n)

            for model in models
                for query_strategy in query_strategies
                    out_dir = split(data_file, '/')[end-1]
                    output_path = "$(exp_dir)results/$(out_dir)"
                    isdir(output_path) || mkdir(output_path)
                    experiment = Dict{Symbol, Any}(
                            :data_file => data_file,
                            :data_set_name => out_dir,
                            :split_strategy_name => data_split_strategy,
                            :initial_pool_strategy_name => initial_pool_strategy,
                            :model => merge(model, Dict(:init_strategy => init_strategy)),
                            :query_strategy => Dict(:type => query_strategy[:type], :param => query_strategy[:param]),
                            :split_strategy => split_strategy,
                            :param => param)
                    exp_hash = hash(sprint(print, experiment))
                    @assert exp_hash == hash(sprint(print, deepcopy(experiment)))
                    experiment[:hash] = "$exp_hash"

                    out_name = splitext(splitdir(data_file)[2])[1]
                    out_name = "$output_path/$(out_name)_$(query_strategy[:type])_$(model[:type])_$(exp_hash).json"
                    experiment[:output_file] = out_name
                    experiment[:log_dir] = "$(exp_dir)log/"
                    push!(experiments, experiment)
                end
            end
        end
    end

    # save the experimental setup
    cp(@__FILE__, "$(exp_dir)$(splitdir(@__FILE__)[2])", follow_symlinks=true)
    cp(config_file, "$(exp_dir)$(splitdir(config_file)[2])", follow_symlinks=true)

    info("Created $exp_dir with $(length(experiments)) instances.")

    open("$(exp_dir)experiment_hashes", "a") do f
        for e in experiments
            write(f, "$(e[:hash])\n")
        end
    end

    save("$(exp_dir)experiments.jld", "experiments", experiments)
end

all(isdir.(data_input_root .* data_dirs)) || error("Not all data dirs are valid.")
data_files = vcat(([data_input_root * x * "/" .*(readdir(data_input_root * x)) for x in data_dirs])...)
println("Found $(length(data_files)) data files.")

for ss in split_strategy
    for ips in initial_pool_strategy
        generate_experiments(data_files, ss, ips, num_resamples_initial_pool)
    end
end
