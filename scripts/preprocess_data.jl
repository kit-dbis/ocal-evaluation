!isempty(ARGS) || error("No config supplied.")
isfile(ARGS[1]) || error("Cannot read '$(ARGS[1])'")
isabspath(ARGS[1]) || error("Please use an absolute path for the config.")
println("Config supplied: '$(ARGS[1])'")
config_file = ARGS[1]
include(config_file)

using MLDataUtils

function process_file(input_file, output_file)
    raw = readdlm(input_file, ',')
    num_attributes = length(find(x -> contains(string(x), "@ATTRIBUTE"), raw[:, 1])) - 2
    id_column = first(find(x -> contains(string(x), "@ATTRIBUTE 'id'"), raw[:, 1])) - 2
    data_start_row = last(find(x -> x == "@DATA", raw[:, 1])) + 1
    raw[:, end] = map(x -> x == "'yes'" ? :outlier : :inlier, raw[:, end])
    data, labels = raw[data_start_row:end, 1:end .!= id_column], raw[data_start_row:end, end]

    @assert size(data, 2) - 1 == num_attributes
    @assert size(data, 1) == length(labels)
    if length(labels) > MAX_VALUES
        p = MAX_VALUES / length(labels)
        (data, labels), _ = stratifiedobs((data, labels), p=p, obsdim=1)
    end
    writedlm(output_file, data, ',')
end

srand(0)
MAX_VALUES = 2000
dataset_dir = "$(data_root)/input/raw/"
output_path = "$(data_root)/input/processed/"
mkpath(output_path)

target_versions = r"withoutdupl_norm_05_v0[1-3]"
for d in data_dirs
    target_files = filter(x -> ismatch(target_versions, x), readdir(dataset_dir * d))
    @assert length(target_files) == 3
    println("[$(d)] Found $(length(target_files)) files.")
    isdir(output_path * d) || mkpath(output_path * d)
    for f in target_files
        process_file(dataset_dir * d * "/" * f, output_path * d * "/" * f[1:end-5] * ".csv")
    end
end
