include("$(@__DIR__)/config.jl")

### data directories ###
data_dirs = ["Annthyroid", "Cardiotocography", "HeartDisease", "Hepatitis", "PageBlocks", "Parkinson", "Pima", "SpamBase", "Stamps"]

### learning scenario ###
initial_pool_strategy = ["Pu", "Pp", "Pn", "Pa"]
split_strategy = ["Sf", "Sh", "Si"]

init_strategy = SimpleCombinedStrategy(RuleOfThumbScott(), BoundedTaxErrorEstimate(0.05, 0.02, 0.98))

#### models ####
models = [Dict(:type => :VanillaSVDD, :param => Dict{Symbol, Any}()),
          Dict(:type => :SVDDneg, :param => Dict{Symbol, Any}()),
          Dict(:type => :SSAD, :param => Dict{Symbol, Any}()),
          Dict(:type => :SSAD, :param => Dict{Symbol, Any}(:κ => 0.5)),
          Dict(:type => :SSAD, :param => Dict{Symbol, Any}(:κ => 0.1))]

#### query strategies ####
query_strategies = [Dict(:type => :MinimumMarginQs, :param => Dict{Symbol, Any}(:p_inlier => 0.05)),
    Dict(:type => :ExpectedMinimumMarginQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :ExpectedMaximumEntropyQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :MinimumLossQs, :param => Dict{Symbol, Any}(:p_inlier => 0.05)),
    Dict(:type => :RandomQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :RandomOutlierQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :HighConfidenceQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :DecisionBoundaryQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :NeighborhoodBasedQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :BoundaryNeighborCombinationQs, :param => Dict{Symbol, Any}())]

num_al_iterations = 50
num_resamples_initial_pool = 1
data_output_root = data_root * "output/evaluation_part1/"
