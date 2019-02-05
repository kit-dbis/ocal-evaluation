remove_and_overwrite = true
validate_package_version = true

JULIA_ENV = joinpath(@__DIR__, "..", "..")
data_root = joinpath(@__DIR__, "..", "..", "data")
data_input_root = joinpath(data_root, "input", "processed")
data_output_root = joinpath(data_root, "output")

worker_list = [("localhost", 1)]
exeflags = `--project="$JULIA_ENV"`
sshflags= `-i path/to/ssh/key/file`

fmt_string = "[{name} | {date} | {level}]: {msg}"
loglevel = "debug"

### data directories ###
data_dirs = Dict("semantic" => ["Arrhythmia", "Annthyroid", "Cardiotocography", "HeartDisease", "Hepatitis", "PageBlocks", "Parkinson", "Pima", "SpamBase", "Stamps"],
                 "literature" => ["ALOI", "Glass", "Ionosphere", "KDDCup99", "Lymphography", "PenDigits", "Shuttle", "WBC", "WDBC", "WPBC", "Waveform"])
