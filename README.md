# ocal-evaluation
_Scripts and notebooks to benchmark one-class active learning strategies._

This repository contains scripts and notebooks to reproduce the experiments and analyses of the paper

> Holger Trittenbach, Adrian Englhardt, Klemens Böhm, "An Overview and a Benchmark of Active Learning for One-Class Classification", DOI: [10.1016/j.eswa.2020.114372](https://doi.org/10.1016/j.eswa.2020.114372), Expert Systems with Applications, 2021.

For more information about this research project, see also the [OCAL project](https://www.ipd.kit.edu/ocal/) website.

## Prerequisites

The experiments are implemented in [Julia](https://julialang.org/), some of the evaluation notebooks are written in python.
This repository contains code to setup the experiments, to execute them, and to analyze the results.
The one-class classifiers and active learning methods are implemented in two separate Julia packages: [SVDD.jl](https://github.com/englhardt/SVDD.jl) and [OneClassActiveLearning.jl](https://github.com/englhardt/OneClassActiveLearning.jl).

The arXiv version 1 paper is based on an older version of this repository (tag `v1.0`).

### Requirements

* Experiments
  * Julia 1.1: `SVDD`, `OneClassActiveLearning`, `Memento`, `MLDataUtils`, `MLKernels`, `JuMP`, `Ipopt`, `CSV`, `Unmarshal`
    (Any other QP Solver that can be used with JuMP works as well. The publication experiments use Gurobi which has free academic licensing.)
* Notebooks
  * Python 3.6 or higher: `matplotlib`, `pandas`, `numpy`, `scipy`, `seaborn`, `json`
  * Julia 1.1: `Plots`, `Colors`, `PyCall`

### Setup
Just clone the repo.
```bash
$ git clone https://github.com/kit-dbis/ocal-evaluation
```

Next, move to the `ocal-evaluation` folder and install all dependencies with the Julia package manager:

```julia
pkg> activate .
pkg> instantiate
```

### Repo Overview

* `data`
  * `input`
    * `raw`: unprocessed data files
    * `processed`: output directory of _preprocess_data.jl_
  * `output`: output directory of experiments; _generate_experiments.jl_ creates the folder structure and experiments; _run_experiments.jl_ writes results and log files
* `notebooks`: jupyter notebooks to analyze experimental results
  * `data-based-QS_visualization`: Figure 5 and Figure 6
  * `evaluation_example.ipynb`: Example 2 (Section 4)
  * `evaluation-part1.ipynb`: Section 4.3
* `scripts`
  * `config`: configuration files for experiments
    * `config.jl`: high-level configuration
    * `config_evaluation_part1.jl`: experiment config for Section 4.3.1 and Section 4.3.4
    * `config_evaluation_part1_qs.jl`: experiment config for Section 4.3.5
    * `config_evaluation_part2.jl`: experiment config for Example 2 (Section 4)
  * `preprocess_data.jl`: preprocess data files into common format
  * `generate_experiments.jl`: generates experiments
  * `reduce_results.jl`: reduces result json files to single result csv
  * `run_experiments`: executes experiments

## Overview

Each step of the benchmark can be reproduced, from the raw data files to the final plots that are presented in the paper.
The benchmark is a pipeline of several dependent processing steps.
Each of the steps can be executed standalone, and takes a well-defined input, and produces a specified output.
The Section [Benchmark Pipeline](#benchmark-pipeline) describes each of the process steps.

Running the benchmark is compute intensive and takes many CPU hours.
Therefore, we also provide the [results to download](https://www.ipd.kit.edu/ocal/ocal-results.zip) (1.2 GB).
This allows to analyze the results (see Step 5) without having to run the whole pipeline.

The code is licensed under a [MIT License](https://github.com/kit-dbis/ocal-evaluation/blob/master/LICENSE.md) and the result data under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
If you use this code or data set in your scientific work, please reference the companion paper.

## Benchmark Pipeline

The benchmark pipeline uses config files to set paths and experiment parameters.
There are two types of config files:
* `scripts/config.jl`: this config defines high-level information on the experiment, such as where the data files are located, and log levels.
* `scripts/config_evaluation_*.jl`: These config files define the experimental grid, including the data sets, classifiers, and active-learning strategies. See Step 2 for a more detailed description. `config_evaluation_part1.jl` is the configuration that has been used for Section 4.3.1 and Section 4.3.4, and `config_evaluation_part1_qs.jl` for Section 4.3.5.

1. _Data Preprocessing_: The preprocessing step transforms publicly available benchmark data sets into a common csv format, and subsamples large data sets to 1000 observations.
   * **Input:** Download [semantic.tar.gz](http://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/semantic.tar.gz) and [literature.tar.gz](http://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/literature.tar.gz) containing the .arff files from the DAMI benchmark repository and extract into `data/input/raw/<literature|semantic>/<data set>` (e.g. `data/input/raw/semantic/Annthyroid/`).
   * **Execution:**
   ```bash
      $ julia --project="." preprocess_data.jl <config.jl>
   ```
   * **Output:** .csv files in `data/input/preprocessed/<data set>`

2. _Generate Experiments_: This step creates a set of experiments. Each experiment in this set is a specific combination of
    * `data set path` (e.g., "data/input/Annthyroid/Annthyroid_withoutdupl_norm_05_v01_r01.csv")
    * `initial pool strategy` (e.g., "Pu")
    * `split strategy` (e.g., "Sf")
    * `model` (e.g., VanillaSVDD)
    * `query strategy` (e.g., DecisionBoundaryPQs)
    * `parameters` (e.g., number of active learning iterations)

   These specific combinations are created as a cross product of the vectors in the config file that is passed as an argument.
   * **Input**: Full path to config file `<config_file.jl>` (e.g., config/config_evaluation_part1.jl), preprocessed data files
   * **Execution:**
   ```bash
    $ julia --project="." generate_experiments.jl <config_file.jl>
   ```
   * **Output:**
     * Creates an experiment directory with the naming `<exp_name>`. The directories created contains several items:
       * `log` directory: skeleton for experiment logs (one file per experiment), and worker logs (one file per worker)
       * `results` directory: skeleton for result files
       * `experiments.jser`: this contains a serialized Julia Array with experiments. Each experiment is a Dict that contains the specific combination. Each experiment can be identified by a unique hash value.
       * `experiment_hashes`: file that contains the hash values of the experiments stored in `experiments.jser`
       * `generate_experiments.jl`: a copy of the file that generated the experiments
       * `config.jl`: a copy of the config file used to generate the experiments

3. _Run Experiments_: This step executes the experiments created in Step 2.
Each experiment is executed on a worker. In the default configuration, a worker is one process on the localhost.
For distributed workers, see Section [Infrastructure and Parallelization](#infractructure-and-parallelization).
A worker takes one specific configuration, runs the active learning experiment, and writes result and log files.
  * **Input:** Generated experiments from step 2.
  * **Execution:**
  ```bash
     $ julia --project="." run_experiments.jl /full/path/to/ocal-evaluation/scripts/config.jl
  ```
  * **Output:** The output files are named by the experiment hash
    * Experiment log (e.g., `data/output/evaluation_part1/log/experiment/10060054773778946468.log`)
    * Result .json file (e.g., `data/output/evaluation_part1/results/Annthyroid/Annthyroid_withoutdupl_norm_05_v01_r01_DecisionBoundaryPQs_SVDDneg_10060054773778946468.json`)

4. _Reduce Results_: Merge of an experiment directory into one .csv by using summary statistics
    * **Input:** Full path to finished experiments.
    * **Execution:**
    ```bash
       $ julia --project="." reduce_results.jl </full/path/to/data/output>
    ```
    * **Output:** A result csv file, `data/output/output.csv`.

5. _Analyze Results:_ jupyter notebooks in the `notebooks`directory to analyze the reduced `.csv`, and individual `.json` files

## Infrastructure and Parallelization

Step 3 _Run Experiments_ can be parallelized over several workers. In general, one can use any [ClusterManager](https://github.com/JuliaParallel/ClusterManagers.jl). In this case, the node that executes `run_experiments.jl` is the driver node. The driver node loads the `experiments.jser`, and initiates a function call for each experiment on one of the workers via `pmap`.

## Authors
We welcome contributions and bug reports.

This package is developed and maintained by [Holger Trittenbach](https://github.com/holtri/) and [Adrian Englhardt](https://github.com/englhardt).
