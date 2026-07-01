# *De novo* design of cell type-specific synthetic enhancers from chromatin accessibility in vertebrate embryos

![plot](./readme_fig.png)

This repository contains code associated with the article [Liu*, Castillo-Hair* et al. *De novo design of cell type-specific synthetic enhancers from chromatin accessibility in vertebrate embryos*](www.example.org). We design synthetic enhancers using **DeepDanio**, our [previously developed](https://doi.org/10.1101/2024.08.27.609971 ) AI predictor of chromatin accessibility trained on scATAC-seq data from zebrafish embryogenesis. A list of cell states predicted by DeepDanio for which enhancers can be designed can be found [here](./src/resources/cell_state_metadata.csv).

## Contents

- **`design/`** - Enhancer design scripts
  - `design.py` - Main sequence design script. It runs Fast SeqProp to generate sequences, saves them along with model predictions, and generates plots.
  - `design_penalize_grhl.py` - Sequence design script that penalizes GRHL motif matches. Used for the EVL motif penalized designs.
  - `design_penalize_mxtx2.py` - Sequence design script that penalizes Mxtx2 motif matches. Used for the YSL motif penalized designs.
  
- **`models/`** - Pre-trained DeepDanio models
  - `download_model_weights.py` - Script to download model weights
  
- **`src/`** - Functions used during sequence design
  - `definitions.py` - Cell state definitions and constants
  - `sequence.py` - Sequence manipulation functions
  - `plot.py` - Plotting and visualization functions
  - `utils.py` - General utility functions
  - `resources/` - Data files.
    - `cell_state_metadata.csv` - Table with all cell states for which enhancers can be designed
    - `edge_prob.txt` - Needed to plot differentiation trajectories with predictions overlaid.

- **`pyproject.toml`** - Project configuration and dependencies

## Requirements

### Hardware requirements
Design scripts require NVIDIA GPUs to run. Enhancers in our publication were designed on `g5.xlarge` EC2 AWS instances, which have an NVIDIA A10G Tensor Core GPU with 24GB VRAM.

### Software requirements

#### OS requirements
This package has been tested on Amazon Linux 2, but most versions of Linux and macOS are expected to be compatible if package requirements are met (see below).

#### Python dependencies
Requirements are listed in `pyproject.toml`. Some important requirements are: Python 3.10 or 3.11, Tensorflow 2.14, Tensorflow probability 0.22.1, and our reimplementation of [Fast SeqProp](https://github.com/castillohair/corefsp).

## Installation guide
We recommend using [uv](https://docs.astral.sh/uv/). To download the repository and install all requirements run the following:

```
git clone https://github.com/castillohair/enhancer-design-zebrafish
cd enhancer-design-zebrafish
uv sync
source .venv/bin/activate
```

This should take care of installing the appropriate GPU-aware version of tensorflow when CUDA drivers are available.

Alternatively, after downloading, install requirements via:
```
pip install -e .
```

## Usage
First download the DeepDanio model weights:

```
cd models
python download_model_weights.py
cd ..
```

Then run `design/design.py` to design sequences. Command line arguments are as follows:

```
cd design
python design.py -h

  --target-idx TARGET_IDX
                        Target cell state index according to src/resources/cell_state_metadata.csv (0-based index)
  --n-seqs N_SEQS       Number of sequences to design (default: 100)
  --model-ensemble-type {pessimistic,average}
                        Model ensemble type (choices: pessimistic, average; default: pessimistic)
  --target-loss-type {percentile,mean}
                        Target loss type (choices: percentile, mean; default: percentile)
  --target-weight TARGET_WEIGHT
                        Weight for maximization of target cell state prediction (default: 1.0)
  --non-target-weight NON_TARGET_WEIGHT
                        Weight for minimization of non-target cell state predictions (default: 1.0)
  --non-target-percentile NON_TARGET_PERCENTILE
                        Percentile for non-target loss (default: 90)
  --output-dir OUTPUT_DIR
                        Output directory to store design results (default: results)
  --output-prefix OUTPUT_PREFIX
                        Prefix for output files (default: auto-generated from target index and cell state)
  --seed SEED           Random seed for sequence initialization. (default: None, random initialization)

```

The scripts `design/design_penalize_grhl.py` and `design/design_penalize_mxtx2.py` have the following additional argument:

```
  --pwm-match-weight PWM_MATCH_WEIGHT
                        Weight for PWM match penalty term (default: 1.0)
```

The outputs of a successful run are as follows:

- `{output_dir}/{output_prefix}_seqs.fasta`: Generated sequences.
- `{output_dir}/{output_prefix}_seqs.png`: Plot with 10 sampled sequences.
- `{output_dir}/{output_prefix}_seq_bitmap.png`: Bitmap representation of generated sequences. Each sequence is one row and the color represents the base.
- `{output_dir}/{output_prefix}_preds_design.csv.gz`: Table with predictions from the ensemble design model used with Fast SeqProp, for each designed sequence.
- `{output_dir}/{output_prefix}_preds_design_boxplot.png`: Distribution of design model predictions.
- `{output_dir}/{output_prefix}_preds_design_trajectory.png`: Median design model predictions overlaid on a differentiation trajectory plot.
- `{output_dir}/{output_prefix}_preds_val.csv.gz`: Table with predictions from the "validation" model (not used with Fast SeqProp) for each designed sequence.
- `{output_dir}/{output_prefix}_preds_val_boxplot.png`: Distribution of validation model predictions.
- `{output_dir}/{output_prefix}_preds_val_trajectory.png`: Median validation model predictions overlaid on a differentiation trajectory plot.
- `{output_dir}/{output_prefix}_4mer_distance.png`: Distribution of 4-mer euclidean distances within pairs of generated sequences, as a measure of sequence diversity.
- `{output_dir}/{output_prefix}_editdistance.png`: Distribution of length-normalized edit distances within pairs of generated sequences, as a measure of sequence diversity.
- `{output_dir}/{output_prefix}_run_metadata.json`: Parameters and other information about this design run.
- `{output_dir}/{output_prefix}_train_history.png`: Plots showing Fast SeqProp loss convergence.

## Reproducing manuscript designs

The enhancer design tasks used to generate the experimentally validated enhancers can be reproduced as follows:

```
# EVL designs in Figure 2. Target cell state is "evl(50epi)".
python design.py --target-idx 22 --n-seqs 10 --model-ensemble-type "average" --target-loss-type "percentile" --target-weight 1.0 --non-target-weight 1.0 --non-target-percentile 90 --output-dir "evl"

# EVL "penalized" designs in Figure 2. Target cell state is "evl(50epi)". Matches to the GRHL motif are penalized.
python design_penalize_grhl.py --target-idx 22 --n-seqs 10 --model-ensemble-type "average" --target-loss-type "percentile" --target-weight 1.0 --non-target-weight 1.0 --non-target-percentile 90 --pwm-match-weight 3.0 --output-dir "evl_penalized_grhl"

# YSL designs in Figure 2. Target cell state is "ysl(shield)".
python design.py --target-idx 24 --n-seqs 10 --model-ensemble-type "average" --target-loss-type "percentile" --target-weight 1.0 --non-target-weight 1.0 --non-target-percentile 90 --output-dir "ysl"

# YSL "penalized" designs in Figure 2. Target cell state is "ysl(shield)". Matches to the Mxtx2 motif are penalized.
python design_penalize_mxtx2.py --target-idx 24 --n-seqs 10 --model-ensemble-type "average" --target-loss-type "percentile" --target-weight 1.0 --non-target-weight 1.0 --non-target-percentile 90 --pwm-match-weight 3.0 --output-dir "ysl_penalized_mxtx2"

# Adaxial cells designs in Figure 3. Target cell state is "adaxial cells(6somite)".
python design.py --target-idx 91 --n-seqs 10 --model-ensemble-type "average" --target-loss-type "percentile" --target-weight 1.0 --non-target-weight 1.0 --non-target-percentile 90 --output-dir "adaxial cells"

# Delayed EVL designs in Figure 4. Target cell state is "evl(6somite)".
python design.py --target-idx 70 --n-seqs 10 --model-ensemble-type "average" --target-loss-type "percentile" --target-weight 1.0 --non-target-weight 1.0 --non-target-percentile 95 --output-dir "evl_delayed"

# Delayed YSL designs in Figure 4. Target cell state is "ysl(6somite)".
python design.py --target-idx 67 --n-seqs 10 --model-ensemble-type "average" --target-loss-type "percentile" --target-weight 1.0 --non-target-weight 1.0 --non-target-percentile 95 --output-dir "ysl_delayed"

```

The design of the 6 somite enhancer "atlas" can be reproduced as follows:
```
python design.py --target-idx TARGET_IDX --n-seqs 50 --model-ensemble-type "pessimistic" --target-loss-type TARGET_LOSS_TYPE --target-weight TARGET_WEIGHT --non-target-weight NON_TARGET_WEIGHT --non-target-percentile NON_TARGET_PERCENTILE --output-dir "6somite_designs"
```
Where all combinations of the following variables were attempted:
- `TARGET_IDX`: all values from 66 to 94, except 67 (ysl) and 70 (evl).
- `TARGET_LOSS_TYPE` as "percentile" or "mean".
- `(TARGET_WEIGHT, NON_TARGET_WEIGHT)` to (1.2, 1.0), (1.1, 1.0), (1.0, 1.0), (1.0, 1.1), (1.0, 1.2), (1.0, 1.5).
- `NON_TARGET_PERCENTILE` to 85, 90, 95, 98 (only if `TARGET_LOSS_TYPE` was "percentile").