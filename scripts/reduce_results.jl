isabspath(ARGS[1]) || error("Please use an absolute path to the output folder.")
println("Reading experiments from: '$(ARGS[1])'")
data_output_root = ARGS[1]

using OneClassActiveLearning, SVDD
using DataFrames, CSV
using Unmarshal, JSON
using Logging, Statistics
using StatsBase

#### Simplify naming

ALIASES = Dict{Symbol, Dict{String, Union{String, Missing}}}(
    :init_strategy_C => Dict("BoundedTaxErrorEstimate" => "TaxBound",
                             "TaxErrorEstimate" => "Tax",
                             "BinarySearchCStrategy" => "BinarySearch",
                             "FixedCStrategy" => "Fix"),
    :init_strategy_gamma  => Dict("FixedGammaStrategy" => "Fix",
                                  "WangGammaStrategy" =>  "Wang",
                                  "WangCombinedInitializationStrategy" => "WangTax",
                                  "RuleOfThumbSilverman" => "Silverman",
                                  "RuleOfThumbScott" => "Scott")
)

function simplify(str, key)
   for (pattern, alias) in ALIASES[key]
       occursin(pattern, str) && return alias
    end
    return str
end

##### Build CSV

array_stats(x) = minimum(x), mean(x), maximum(x)
prefix_named_tuples(x, prefix) = merge(NamedTuple(), [Symbol("$(prefix)_$k")=>v for (k, v) in pairs(x)])

function get_time_mem_stats(r::OneClassActiveLearning.Result)
    if !haskey(r.al_history, :time_fit) || !haskey(r.al_history, :mem_fit) ||
       !haskey(r.al_history, :time_qs) || !haskey(r.al_history, :mem_qs)
        return (time_fit_min=missing, time_fit_mean=missing, time_fit_max=missing,
                mem_fit_min=missing, mem_fit_mean=missing, mem_fit_max=missing,
                time_qs_min=missing, time_qs_mean=missing, time_qs_max=missing,
                mem_qs_min=missing, mem_qs_mean=missing, mem_qs_max=missing)
    end
    total_run_time = r.al_summary[:runtime][:time_exp]
    time_fit_min, time_fit_mean, time_fit_max = array_stats(values(r.al_history, :time_fit))
    mem_fit_min, mem_fit_mean, mem_fit_max = array_stats(values(r.al_history, :mem_fit))
    time_qs_min, time_qs_mean, time_qs_max = array_stats(values(r.al_history, :time_qs))
    mem_qs_min, mem_qs_mean, mem_qs_max = array_stats(values(r.al_history, :mem_qs))
    return (total_run_time=total_run_time,
            time_fit_min=time_fit_min, time_fit_mean=time_fit_mean, time_fit_max=time_fit_max,
            mem_fit_min=mem_fit_min, mem_fit_mean=mem_fit_mean, mem_fit_max=mem_fit_max,
            time_qs_min=time_qs_min, time_qs_mean=time_qs_mean, time_qs_max=time_qs_max,
            mem_qs_min=mem_qs_min, mem_qs_mean=mem_qs_mean, mem_qs_max=mem_qs_max)
end

function get_al_summary(r::OneClassActiveLearning.Result, metric::Symbol, metric_short_name::String, n=5, stability_window=10)
    if !haskey(r.al_summary, metric)
        summary = (start_quality=missing, end_quality=missing, maximum=missing, ramp_up=missing, quality_range=missing, total_quality_range=missing,
                   average_end_quality=missing, average_quality_change=missing, average_gain=missing, average_loss=missing,
                   learning_stability=missing, ratio_of_outlier_queries=missing,
                   aulc=missing, reyes_paulc=missing, reyes_naulc=missing, reyes_tpr=missing, reyes_tnr=missing, reyes_tp=missing)
    else
        cur_data = r.al_summary[metric]
        start_quality = cur_data[:start_quality]
        end_quality = cur_data[:end_quality]
        maximum = cur_data[:maximum]
        ramp_up = cur_data[:ramp_up][n]
        quality_range = cur_data[:quality_range]
        total_quality_range = cur_data[:total_quality_range]
        average_end_quality = if length(cur_data[:average_end_quality]) - n + 1 > 0
                                    cur_data[:average_end_quality][end-n+1]
                              else missing end
        average_quality_change = cur_data[:average_quality_change]
        average_gain = cur_data[:average_gain]
        average_loss = cur_data[:average_loss]
        learning_stability = if length(cur_data[:learning_stability]) - stability_window + 1 > 0
                                    cur_data[:learning_stability][end-stability_window+1]
                              else missing end
        ratio_of_outlier_queries = cur_data[:ratio_of_outlier_queries]
        aulc_based_stats = if :aulc in keys(cur_data)
            aulc = cur_data[:aulc]
            reyes_paulc = cur_data[:reyes_paulc]
            reyes_naulc = cur_data[:reyes_naulc]
            reyes_tpr = cur_data[:reyes_tpr]
            reyes_tnr = cur_data[:reyes_tnr]
            reyes_tp = cur_data[:reyes_tp]
            (aulc=aulc, reyes_paulc=reyes_paulc, reyes_naulc=reyes_naulc, reyes_tpr=reyes_tpr, reyes_tnr=reyes_tnr, reyes_tp=reyes_tp)
        else
            (aulc=missing, reyes_paulc=missing, reyes_naulc=missing, reyes_tpr=missing, reyes_tnr=missing, reyes_tp=missing)
        end
        summary = (start_quality=start_quality, end_quality=end_quality, maximum=maximum, ramp_up=ramp_up, quality_range=quality_range, total_quality_range=total_quality_range,
                   average_end_quality=average_end_quality, average_quality_change=average_quality_change, average_gain=average_gain, average_loss=average_loss,
                   learning_stability=learning_stability, ratio_of_outlier_queries=ratio_of_outlier_queries, aulc_based_stats...)
    end
    return prefix_named_tuples(summary, metric_short_name)
end

function calc_stats(r::OneClassActiveLearning.Result)
    id = string(r.id)
    dir, file_name = splitdir(normpath(r.experiment[:data_file]))
    data_set = r.experiment[:data_set_name]
    split_strategy = r.experiment[:split_strategy_name][1]
    initial_pool_strategy = r.experiment[:initial_pool_strategy_name][1]
    initial_pool_resample_version = r.experiment[:param][:initial_pool_resample_version]
    initial_pool_num_labeled = sum(r.experiment[:param][:initial_pools] .!= "U")
    model = r.experiment[:model][:type]
    if model == "SSAD"
        kappa = get(r.experiment[:model][:param], :κ, 1.0)
        model = "$(model)_$(kappa)"
    end
    init_strategy_C = simplify(r.experiment[:model][:init_strategy], :init_strategy_C)
    init_strategy_gamma = simplify(r.experiment[:model][:init_strategy], :init_strategy_gamma)
    init_strategy_gamma_fitted_value = if haskey(r.experiment[:model], :fitted) && haskey(r.experiment[:model][:fitted], :kernel)
            first(match(r"\((.*)\)", r.experiment[:model][:fitted][:kernel]).captures)
        else missing end

    qs = split(string(r.experiment[:query_strategy][:type]), ".")[end]

    num_points = r.data_stats.num_observations
    num_dimensions = r.data_stats.num_dimensions
    num_al_iterations = r.experiment[:param][:num_al_iterations]

    exit_code = string(r.status[:exit_code])
    res = (id=id, file_name=file_name, data_set=data_set, split_strategy=split_strategy, initial_pool_strategy=initial_pool_strategy,
            initial_pool_resample_version=initial_pool_resample_version, initial_pool_num_labeled=initial_pool_num_labeled, model=model,
            init_strategy_C=init_strategy_C, init_strategy_gamma=init_strategy_gamma,
            init_strategy_gamma_fitted_value=init_strategy_gamma_fitted_value,
            qs=qs,
            num_points=num_points, num_dimensions=num_dimensions, num_al_iterations=num_al_iterations, exit_code=exit_code)
    stats = reduce(merge, [get_time_mem_stats(r),
            get_al_summary(r, :matthews_corr, "m"),
            get_al_summary(r, :cohens_kappa, "c"),
            get_al_summary(r, :auc, "auc"),
            get_al_summary(r, Symbol("auc_fpr_normalized_0.05"), "pauc")])
    return merge(res, stats)
end

function reduce_results(input_path)
    df = Any[]
    for scenario in readdir(input_path)
        if occursin(".csv", scenario)
            continue
        end
        for ds in readdir(joinpath(input_path, scenario, "results"))
            @info ds
            result_path = joinpath(input_path, scenario, "results", ds)
            result_files = joinpath.(result_path, filter(x -> !startswith(x, "."), readdir(result_path)))
            for r in result_files
                @info r
                res = Unmarshal.unmarshal(OneClassActiveLearning.Result, JSON.parsefile(r))
                push!(df,  merge((scenario=scenario,), calc_stats(res)))
            end
        end
    end
    df_types = mapreduce(x -> [typeof(t) for t in x], (d1, d2) -> [Union{t1,t2} for (t1,t2) in zip(d1, d2)], df)
    df_names = keys(df[1])
    res_df = DataFrame(df_types, [df_names...], 0)
    for row in df
        push!(res_df, row)
    end
    output_file = joinpath(input_path, "$(split(input_path, "/")[end]).csv")
    @info "Writing result to '$(output_file)'"
    CSV.write(output_file, res_df)
    return res_df
end

reduce_results(data_output_root)
