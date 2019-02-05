!isempty(ARGS) || error("No config supplied.")
isfile(ARGS[1]) || error("Cannot read '$(ARGS[1])'")
isabspath(ARGS[1]) || error("Please use an absolute path for the config.")
println("Config supplied: '$(ARGS[1])'")
config_file = ARGS[1]
include(config_file)

using MLDataUtils
using Random
using DelimitedFiles

function downsample_class(data, labels, target_class, other_class, target_percentage)
    target_indicies = shuffle!(findall(x -> x .== target_class, labels))
    mask = falses(length(labels))
    mask[labels .== other_class] .= true
    mask[target_indicies[1:ceil(Int, target_percentage * sum(labels .== other_class) / (1 - target_percentage))]] .= true
    return data[mask, :], labels[mask]
end

function process_file(input_file, output_file; num_versions=1)
    raw = readdlm(input_file, ',')
    num_attributes = length(findall(x -> occursin("@ATTRIBUTE", string(x)), raw[:, 1])) - 2
    id_column = findfirst(x -> occursin("@ATTRIBUTE 'id'", string(x)), raw[:, 1]) - 1
    label_column = findfirst(x -> occursin("@ATTRIBUTE 'outlier'", string(x)), raw[:, 1]) - 1
    data_start_row = findlast(x -> x == "@DATA", raw[:, 1]) + 1
    raw[:, label_column] = map(x -> x == "'yes'" ? :outlier : :inlier, raw[:, label_column])
    data, labels = raw[data_start_row:end, [i for i in 1:size(raw, 2) if i != id_column && i != label_column]], raw[data_start_row:end, label_column]
    data = hcat(data, labels)
    @assert size(data, 2) - 1 == num_attributes
    @assert size(data, 1) == length(labels)

    for i in 1:num_versions
        Random.seed!(i)
        outlier_percentage = sum(labels .== :outlier) / length(labels)
        resampling = outlier_percentage != TARGET_OUTLIER_PERCENTAGE || length(labels) > MAX_VALUES
        target_output_file = resampling ? "$(output_file[1:end-4])_r0$i.csv" : output_file
        @info "Generating '$target_output_file'."
        res_data, res_labels = copy(data), copy(labels)
        if outlier_percentage > TARGET_OUTLIER_PERCENTAGE
            @info "Downsampling outlier class (outlier_percentage = $(outlier_percentage))."
            res_data, res_labels = downsample_class(res_data, res_labels, :outlier, :inlier, TARGET_OUTLIER_PERCENTAGE)
        elseif outlier_percentage < TARGET_OUTLIER_PERCENTAGE
            @info "Downsampling inlier class (outlier_percentage = $(outlier_percentage))."
            res_data, res_labels = downsample_class(res_data, res_labels, :inlier, :outlier, 1 - TARGET_OUTLIER_PERCENTAGE)
        end
        if length(res_labels) > MAX_VALUES
            @info "Downsampling from $(length(res_labels)) to $MAX_VALUES observations."
            p = MAX_VALUES / length(res_labels)
            (res_data, res_labels), _ = stratifiedobs((res_data, res_labels), p=p, obsdim=1)
        end
        outlier_percentage = sum(res_labels .== :outlier) / length(res_labels)
        @info "Final outlier_percentage = $(outlier_percentage))."
        @assert size(res_data, 1) == length(res_labels)
        @assert abs(outlier_percentage - TARGET_OUTLIER_PERCENTAGE) < 0.01
        writedlm(target_output_file, res_data, ',')
    end
end

Random.seed!(0)
MAX_VALUES = 1000
TARGET_OUTLIER_PERCENTAGE = 0.05
target_versions_semantic = r"withoutdupl_norm_05_v0[1-3]"
target_versions_literature = r"withoutdupl_norm"
dataset_dir = normpath(joinpath(data_root, "input", "raw"))
output_path = normpath(joinpath(data_root, "input", "processed"))

mkpath(output_path)
@info "Saving processed files to $output_path."

for dataset_class in ["semantic", "literature"]
    for d in data_dirs[dataset_class]
        @info d
        outdir = joinpath(output_path, d)
        isdir(outdir) || mkpath(outdir)
        if dataset_class == "semantic"
            target_files = filter(x -> occursin(target_versions_semantic, x), readdir(joinpath(dataset_dir, dataset_class, d)))
            @assert length(target_files) == 3
            @info "[$(d)] Found $(length(target_files)) files."
            for f in target_files
               process_file(joinpath(dataset_dir, "semantic", d, f), joinpath(outdir, f[1:end-5] * ".csv"))
            end
        else
            target_files = filter(x -> occursin(target_versions_literature, x), readdir(joinpath(dataset_dir, dataset_class, d)))
            if (i = findfirst(x -> occursin("catremoved", x), target_files)) !== nothing
                target_file = target_files[i]
            else
                target_file = first(target_files)
            end
            @info "[$(d)] Resampling '$target_file'."
            process_file(joinpath(dataset_dir, dataset_class, d, target_file), joinpath(outdir, target_file[1:end-5] * ".csv"), num_versions=3)
        end
    end
end
