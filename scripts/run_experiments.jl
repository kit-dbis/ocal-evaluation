# Command line args
!isempty(ARGS) || error("No config supplied.")
isfile(ARGS[1]) || error("Cannot read '$(ARGS[1])'")
isabspath(ARGS[1]) || error("Please use an absolute path for the config.")
println("Config supplied: '$(ARGS[1])'")
using SVDD, OneClassActiveLearning
include(ARGS[1])
for s in readdir(data_output_root)
    experiment_directory = data_output_root * s
    println("Experiment directory '$experiment_directory'")
    isdir(experiment_directory) || error("Experiment directory not found '$experiment_directory'.")
end

# Load packages
using JLD

# setup workers
localhost = filter(x -> x[1] == "localhost", worker_list)
remote_servers = filter(x -> x[1] != "localhost", worker_list)

length(localhost) > 0 && addprocs(localhost[1][2])
length(remote_servers) > 0 && addprocs(remote_servers, sshflags=sshflags)

# validate package versions
@everywhere function get_package_commit(pkg_name; pkg_home = "$(Pkg.dir())")
    pkg_path = "$(pkg_home)/$(pkg_name)"
    cmd = `git -C $(pkg_path) rev-parse HEAD`
    (gethostname(), strip(string(readstring(cmd))))
end

function check_all_packages(worker_ids, packages::Associative)
    for (name, local_githash) in packages
        for id in worker_ids
            remote_version = remotecall_fetch(get_package_commit, id, name)
            @assert remote_version[2] == local_githash "Host: $(remote_version[1]) has version mismatch for package $(name)."
        end
    end
    return true
end

if validate_package_version
    packages = Dict("SVDD" => get_package_commit("SVDD")[2],
                    "OneClassActiveLearning" => get_package_commit("OneClassActiveLearning")[2])
    check_all_packages(workers(), packages) && info("Package versions validated.")
else
    warn("CAUTION: Package validation is currently set to false. Set 'validate_package_version = true' in $(config_file) if you run production experiments.")
end

# Load remote packages and functions
info("Loading packages on all workers.")
@everywhere using SVDD, OneClassActiveLearning, JSON, Ipopt, Memento, Gurobi
@everywhere import SVDD: RandomOCClassifier

@everywhere fmt_string = "[{name} | {date} | {level}]: {msg}"
@everywhere loglevel = "debug"

@everywhere function setup_logging(experiment)
    setlevel!(getlogger("root"), "error")
    setlevel!(getlogger(OneClassActiveLearning), loglevel)
    setlevel!(getlogger(SVDD), loglevel)

    exp_logfile = "$(experiment[:log_dir])experiment/$(experiment[:hash]).log"
    worker_logfile = "$(experiment[:log_dir])worker/$(gethostname())_$(getpid()).log"

    WORKER_LOGGER = Memento.config!("runner", "debug"; fmt=fmt_string)

    exp_handler = DefaultHandler(exp_logfile, DefaultFormatter(fmt_string))
    push!(getlogger(OneClassActiveLearning), exp_handler, experiment[:hash])
    push!(getlogger(SVDD), exp_handler, experiment[:hash])
    push!(WORKER_LOGGER, exp_handler, experiment[:hash])

    worker_handler = DefaultHandler(worker_logfile, DefaultFormatter(fmt_string))
    setlevel!(gethandlers(WORKER_LOGGER)["console"], "error")
    push!(WORKER_LOGGER, worker_handler)

    return WORKER_LOGGER
end

@everywhere function run_experiment(experiment::Dict)
    # Make experiments deterministic
    srand(0)

    WORKER_LOGGER = setup_logging(experiment)
    info(WORKER_LOGGER, "host: $(gethostname()) worker: $(getpid()) exp_hash: $(experiment[:hash])")
    if isfile(experiment[:output_file])
        warn(WORKER_LOGGER, "Aborting experiment because the output file already exists. Filename: $(experiment[:output_file])")
        return nothing
    end

    res = Result(experiment)
    errorfile = "$(experiment[:log_dir])worker/$(gethostname())_$(getpid())"
    try
        time_exp = @elapsed res = OneClassActiveLearning.active_learn(experiment)
        res.al_summary[:runtime] = Dict(:time_exp => time_exp)
    catch e
        res.status[:exit_code] = Symbol(typeof(e))
        warn(WORKER_LOGGER, "Experiment $(experiment[:hash]) finished with unkown error.")
        warn(WORKER_LOGGER, e)
    finally
        if res.status[:exit_code] != :success
            info(WORKER_LOGGER, "Writing error hash to $errorfile.error.")
            open("$errorfile.error", "a") do f
                print(f, "$(experiment[:hash])\n")
            end
        end
        info(WORKER_LOGGER, "Writing result to $(experiment[:output_file]).")
        OneClassActiveLearning.write_result_to_file(experiment[:output_file], res)
    end
    delete!(gethandlers(getlogger(OneClassActiveLearning)), experiment[:hash])
    delete!(gethandlers(getlogger(SVDD)), experiment[:hash])
    delete!(gethandlers(WORKER_LOGGER), experiment[:hash])
    return nothing
end

# load and run experiments
all_experiments = []
for s in readdir(data_output_root)
    info("Running experiments in directory $s")
    exp_dir = data_output_root * s * "/"
    info("Loading experiments.jld")
    # load experiments
    experiments = JLD.load("$(exp_dir)experiments.jld", "experiments")
    append!(all_experiments, experiments)
end
info("Running $(length(all_experiments)) experiments.")
info("Running experiments...")
pmap(run_experiment, all_experiments)
info("Done.")

# cleanup
rmprocs(workers())
