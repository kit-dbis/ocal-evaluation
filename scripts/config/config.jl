remove_and_overwrite = true
validate_package_version = true

data_root = "$(@__DIR__)/../../data/"
data_input_root = data_root * "input/processed/"
data_output_root = data_root * "output/"

worker_list = [("localhost", 1)]
#sshflags= `-i path/to/ssh/key/file`

fmt_string = "[{name} | {date} | {level}]: {msg}"
loglevel = "debug"

### data directories ###
data_dirs = ["Annthyroid", "Cardiotocography", "HeartDisease", "Hepatitis", "PageBlocks", "Parkinson", "Pima", "SpamBase", "Stamps"]
data_info = "3f-2000n-5p"
