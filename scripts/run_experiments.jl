# Command line args
!isempty(ARGS) || error("No config supplied.")
isfile(ARGS[1]) || error("Cannot read '$(ARGS[1])'")
isabspath(ARGS[1]) || error("Please use an absolute path for the config.")
println("Config supplied: '$(ARGS[1])'")
using SVDD, OneClassActiveLearning
include(ARGS[1])

# Load packages
using Serialization, Distributed, Logging

# setup workers
localhost = filter(x -> x[1] == "localhost", worker_list)
remote_servers = filter(x -> x[1] != "localhost", worker_list)

length(localhost) > 0 && addprocs(localhost[1][2], exeflags=exeflags)
length(remote_servers) > 0 && addprocs(remote_servers, sshflags=sshflags, exeflags=exeflags)

# validate package versions
@everywhere function get_git_hash(path)
    cmd = `git -C $path rev-parse HEAD`
    (gethostname(), strip(read(cmd, String)))
end

function setup_julia_environment()
    local_githash = get_git_hash(JULIA_ENV)[2]
    for id in workers()
        remote_name, remote_githash = remotecall_fetch(get_git_hash, id, JULIA_ENV)
        @assert remote_githash == local_githash "Host: $remote_name has version mismatch." *
                                                    "Hash is '$remote_githash' instead of '$local_githash'."
    end
end

# Load remote packages and functions
@info "Loading packages on all workers."
@everywhere using Pkg
setup_julia_environment()
@everywhere using SVDD, OneClassActiveLearning, Memento, Gurobi, Random
@everywhere import SVDD: RandomOCClassifier

@everywhere fmt_string = "[{name} | {date} | {level}]: {msg}"
@everywhere loglevel = "debug"

@everywhere function setup_logging(experiment)
    setlevel!(getlogger("root"), "error")
    setlevel!(getlogger(OneClassActiveLearning), loglevel)
    setlevel!(getlogger(SVDD), loglevel)

    exp_logfile = joinpath(experiment[:log_dir], "experiment", "$(experiment[:hash]).log")
    worker_logfile = joinpath(experiment[:log_dir], "worker", "$(gethostname())_$(getpid()).log")

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

@everywhere function cleanup_logging(worker_logger::Logger, experiment_hash)
    delete!(gethandlers(getlogger(OneClassActiveLearning)), experiment_hash)
    delete!(gethandlers(getlogger(SVDD)), experiment_hash)
    delete!(gethandlers(worker_logger), experiment_hash)
    return nothing
end

@everywhere function Memento.warn(logger::Logger, error::Gurobi.GurobiError)
    Memento.warn(logger, "GurobiError(exit_code=$(error.code), msg='$(error.msg)')")
end

@everywhere function Memento.debug(logger::Logger, error::Gurobi.GurobiError)
    Memento.warn(logger, "GurobiError(exit_code=$(error.code), msg='$(error.msg)')")
end

@everywhere function Memento.debug(logger::Logger, error::ErrorException)
    Memento.warn(logger, "Caught ErrorException, msg='$(error.msg)')")
end

@everywhere function run_experiment(experiment::Dict)
    # Make experiments deterministic
    Random.seed!(0)

    WORKER_LOGGER = setup_logging(experiment)
    info(WORKER_LOGGER, "host: $(gethostname()) worker: $(getpid()) exp_hash: $(experiment[:hash])")
    if isfile(experiment[:output_file])
        warn(WORKER_LOGGER, "Aborting experiment because the output file already exists. Filename: $(experiment[:output_file])")
        cleanup_logging(WORKER_LOGGER, experiment[:hash])
        return nothing
    end

    res = Result(experiment)
    errorfile = joinpath(experiment[:log_dir], "worker", "$(gethostname())_$(getpid())")
    try
        time_exp = @elapsed res = OneClassActiveLearning.active_learn(experiment)
        res.al_summary[:runtime] = Dict(:time_exp => time_exp)
    catch e
        res.status[:exit_code] = Symbol(typeof(e))
        @warn "Experiment $(experiment[:hash]) finished with unkown error."
        @warn e
        @warn stacktrace(catch_backtrace())
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
        cleanup_logging(WORKER_LOGGER, experiment[:hash])
    end

    return nothing
end

# load and run experiments
all_experiments = []
for s in readdir(data_output_root)
    if occursin(".csv", s)
        continue
    end
    @info "Running experiments in directory $s"
    exp_dir = joinpath(data_output_root, s)
    @info "Loading experiments.jld"
    # load experiments
    experiments = deserialize(open(joinpath(exp_dir, "experiments.jser")))
    append!(all_experiments, experiments)
end
@info "Running $(length(all_experiments)) experiments."
@info "Running experiments..."
pmap(run_experiment, all_experiments)
@info "Done."

# cleanup
rmprocs(workers())
