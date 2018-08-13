
!isempty(ARGS) || error("No config supplied.")
isfile(ARGS[1]) || error("Cannot read '$(ARGS[1])'")
isabspath(ARGS[1]) || error("Please use an absolute path for the config.")
println("Config supplied: '$(ARGS[1])'")
config_file = ARGS[1]

using OneClassActiveLearning
using SVDD
using DataFrames
using CSV

array_stats(x) = minimum(x), mean(x), maximum(x)
init_strategy_names = Dict("SVDD.SimpleCombinedStrategy(SVDD.FixedGammaStrategy(SquaredExponentialKernel(2.0)), SVDD.FixedCStrategy(0.03))" => "Fix_Fix",
                           "SVDD.SimpleCombinedStrategy(SVDD.FixedGammaStrategy(SquaredExponentialKernel(2.0)), SVDD.BoundedTaxErrorEstimate(0.05, 0.02, 0.98))" => "Fix_Tax",
                           "SVDD.SimpleCombinedStrategy(SVDD.RuleOfThumbScott(), SVDD.BoundedTaxErrorEstimate(0.05, 0.02, 0.98))" => "Thumb_Tax",
                           "SVDD.GammaFirstCombinedStrategy(SVDD.RuleOfThumbScott(), SVDD.BinarySearchCStrategy(0.05, 0.5, 5, 0.01, Gurobi.GurobiSolver(nothing, Any[(:OutputFlag, 0), (:Threads, 1)])))" => "Thumb_Bin")

function get_time_mem_stats(r::OneClassActiveLearning.Result)
    time_data_min, time_data_mean, time_data_max = array_stats(values(r.al_history, :time_set_data))
    time_fit_min, time_fit_mean, time_fit_max = array_stats(values(r.al_history, :time_fit))
    mem_fit_min, mem_fit_mean, mem_fit_max = array_stats(values(r.al_history, :mem_fit))
    time_qs_min, time_qs_mean, time_qs_max = array_stats(values(r.al_history, :time_qs))
    mem_qs_min, mem_qs_mean, mem_qs_max = array_stats(values(r.al_history, :mem_qs))
    return (time_data_min, time_data_mean, time_data_max,
            time_fit_min, time_fit_mean, time_fit_max, mem_fit_min, mem_fit_mean, mem_fit_max,
            time_qs_min, time_qs_mean, time_qs_max, mem_qs_min, mem_qs_mean, mem_qs_max)
end

function get_al_summary(r::OneClassActiveLearning.Result, metric::Symbol, n=5, stability_window=10)
    start_quality = r.al_summary[metric][:start_quality]
    end_quality = r.al_summary[metric][:end_quality]
    maximum = r.al_summary[metric][:maximum]
    ramp_up = r.al_summary[metric][:ramp_up][n]
    quality_range = r.al_summary[metric][:quality_range]
    total_quality_range = r.al_summary[metric][:total_quality_range]
    average_end_quality = r.al_summary[metric][:average_end_quality][end-n+1]
    average_quality_change = r.al_summary[metric][:average_quality_change]
    average_gain = r.al_summary[metric][:average_gain]
    average_loss = r.al_summary[metric][:average_loss]
    learning_stability = r.al_summary[metric][:learning_stability][end-stability_window+1]
    ratio_of_outlier_queries = r.al_summary[metric][:ratio_of_outlier_queries]
    return (start_quality, end_quality, maximum, ramp_up, quality_range, total_quality_range,
            average_end_quality, average_quality_change, average_gain, average_loss,
            learning_stability, ratio_of_outlier_queries)
end

function calc_stats(r::OneClassActiveLearning.Result)
    id = string(r.id)
    file_name = r.experiment[:data_file]
    data_set = r.experiment[:data_set_name]
    split_strategy = r.experiment[:split_strategy_name]
    initial_pool_strategy = r.experiment[:initial_pool_strategy_name]
    initial_pool_resample_version = r.experiment[:param][:initial_pool_resample_version]
    model = r.experiment[:model][:type]
    if model == "SSAD" && haskey(r.experiment[:model][:param], :κ)
        model = "$(model)_$(r.experiment[:model][:param][:κ])"
    end
    init_strategy = init_strategy_names[r.experiment[:model][:init_strategy]]
    qs = split(string(r.experiment[:query_strategy][:type]), ".")[end]
    num_points = r.data_stats.num_observations
    num_dimensions = r.data_stats.num_dimensions
    exit_code = string(r.status[:exit_code])
    if exit_code == "success"
        return (id, file_name, data_set, split_strategy, initial_pool_strategy,
                initial_pool_resample_version, model, init_strategy, qs,
                num_points, num_dimensions, exit_code,
                get_time_mem_stats(r)...,
                get_al_summary(r, :matthews_corr)...,
                get_al_summary(r, :cohens_kappa)...,
                get_al_summary(r, :auc)...,
                get_al_summary(r, Symbol("auc_fpr_normalized_0.05"))...)
    else
        return (id, file_name, data_set, split_strategy, initial_pool_strategy,
                initial_pool_resample_version, model, init_strategy, qs,
                num_points, num_dimensions, exit_code,
                fill(0.0, 63)...)
    end
end

function reduce_results(input_path)
    df = DataFrame(id=String[], file_name=String[], data_set=String[], split_strategy=String[],
        initial_pool_strategy=String[], initial_pool_resample_version=Float64[],
        model=String[], init_strategy=String[], qs=String[],
        num_points=Float64[], num_dimensions=Float64[], exit_code=String[],
        time_data_min=Float64[], time_data_mean=Float64[], time_data_max=Float64[],
        time_fit_min=Float64[], time_fit_mean=Float64[], time_fit_max=Float64[],
        mem_fit_min=Float64[], mem_fit_mean=Float64[], mem_fit_max=Float64[],
        time_qs_min=Float64[], time_qs_mean=Float64[], time_qs_max=Float64[],
        mem_qs_min=Float64[], mem_qs_mean=Float64[], mem_qs_max=Float64[],
        m_start_quality=Float64[], m_end_quality=Float64[], m_maximum=Float64[], m_ramp_up=Float64[], m_quality_range=Float64[], m_total_quality_range=Float64[],
        m_average_end_quality=Float64[], m_average_quality_change=Float64[], m_average_gain=Float64[], m_average_loss=Float64[], m_learning_stability=Float64[], m_ratio_of_outlier_queries=Float64[],
        k_start_quality=Float64[], k_end_quality=Float64[], k_maximum=Float64[], k_ramp_up=Float64[], k_quality_range=Float64[], k_total_quality_range=Float64[],
        k_average_end_quality=Float64[], k_average_quality_change=Float64[], k_average_gain=Float64[], k_average_loss=Float64[], k_learning_stability=Float64[], k_ratio_of_outlier_queries=Float64[],
        auc_start_quality=Float64[], auc_end_quality=Float64[], auc_maximum=Float64[], auc_ramp_up=Float64[], auc_quality_range=Float64[], auc_total_quality_range=Float64[],
        auc_average_end_quality=Float64[], auc_average_quality_change=Float64[], auc_average_gain=Float64[], auc_average_loss=Float64[], auc_learning_stability=Float64[], auc_ratio_of_outlier_queries=Float64[],
        pauc_start_quality=Float64[], pauc_end_quality=Float64[], pauc_maximum=Float64[], pauc_ramp_up=Float64[], pauc_quality_range=Float64[], pauc_total_quality_range=Float64[],
        pauc_average_end_quality=Float64[], pauc_average_quality_change=Float64[], pauc_average_gain=Float64[], pauc_average_loss=Float64[], pauc_learning_stability=Float64[], pauc_ratio_of_outlier_queries=Float64[])

    for scenario in readdir(input_path)
        if contains(scenario, ".csv")
            continue
        end
        for ds in readdir(input_path * scenario * "/results/")
            println(ds)
            result_path = input_path * scenario * "/results/" * ds * "/"
            result_files = result_path .* filter(x -> !startswith(x, "."), readdir(result_path))
            for r in result_files
                println(r)
                res = Unmarshal.unmarshal(OneClassActiveLearning.Result, JSON.parsefile(r))
                push!(df, calc_stats(res))
            end
        end
    end
    output_file = "$(input_path)/$(split(input_path, "/")[end-1]).csv"
    info("Writing result to '$(output_file)'")
    CSV.write(output_file, df)
    return df
end

include(config_file)
reduce_results(data_output_root)
