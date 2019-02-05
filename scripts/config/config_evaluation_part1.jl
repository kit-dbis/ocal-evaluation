include(joinpath(@__DIR__, "config.jl"))
using JuMP, Gurobi, MLKernels

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)

### data directories ###
data_dirs = ["ALOI", "Annthyroid", "Arrhythmia", "Cardiotocography", "Glass",
             "HeartDisease", "Hepatitis", "Ionosphere", "KDDCup99", "Lymphography",
             "PageBlocks", "Parkinson", "PenDigits", "Pima", "Shuttle", "SpamBase",
             "Stamps", "WBC", "WDBC", "WPBC", "Waveform"]

### learning scenario ###
initial_pool_strategy = [("Pu", Dict()), ("Pp", Dict()), ("Pn", Dict(:n => 25)), ("Pa", Dict())]
split_strategy = [(x, Dict()) for x in ["Sf", "Sh", "Si"]]

init_strategies = [SimpleCombinedStrategy(RuleOfThumbScott(), BoundedTaxErrorEstimate(0.05, 0.02, 0.98)),
                   WangCombinedInitializationStrategy(solver, 2.0.^range(-4, stop=4, step=1.0), BoundedTaxErrorEstimate(0.05, 0.02, 0.98))]

#### models ####
models = [Dict(:type => :VanillaSVDD, :param => Dict{Symbol, Any}()),
          Dict(:type => :SVDDneg, :param => Dict{Symbol, Any}()),
          Dict(:type => :SSAD, :param => Dict{Symbol, Any}()),
          Dict(:type => :SSAD, :param => Dict{Symbol, Any}(:κ => 0.5)),
          Dict(:type => :SSAD, :param => Dict{Symbol, Any}(:κ => 0.1))]
classify_precision = SVDD.OPT_PRECISION

#### oracle ####
oracle_param = Dict{Symbol, Any}(
    :type => PoolOracle,
    :param => Dict{Symbol, Any}()
)

#### query strategies ####
query_strategies = [Dict(:type => :MinimumMarginPQs, :param => Dict{Symbol, Any}(:p_inlier => 0.05)),
    Dict(:type => :ExpectedMinimumMarginPQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :ExpectedMaximumEntropyPQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :MinimumLossPQs, :param => Dict{Symbol, Any}(:p_inlier => 0.05)),
    Dict(:type => :RandomPQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :RandomOutlierPQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :HighConfidencePQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :DecisionBoundaryPQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :NeighborhoodBasedPQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :BoundaryNeighborCombinationPQs, :param => Dict{Symbol, Any}())]

num_al_iterations = 50
num_resamples_initial_pool = 1
exp_name = "evaluation_part1"
