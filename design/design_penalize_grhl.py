"""
Generate DNA sequences with chromatin accessibility specific to a target cell state, using
the DeepDanio model and Fast SeqProp as the optimization method, and penalizing occurrences
of the GRHL motif.

Script sections:
1. Imports and constants: Import necessary libraries and define paths to model files and
   metadata.
2. Loss functions: Specify sequence features to optimize. We define target loss functions
   that maximizes the predicted target cell state accessibility - transformed via a square
   root function to obtain diminishing returns for really high prediction values - and
   minimize either the average or a specified percentile of non-target predictions. Higher
   non-target percentile corresponds to more stringent designs but may lead to lower target
   activity. Additionally, we define a PWM loss function that penalizes similarity to a
   given PWM, and a PWM loss against repeated bases.
3. Main sequence design function: Loads models and metadata, runs Fast SeqProp to generate
   sequences, calculates predictions, saves results, and generates plots. We can use either
   a pessimistic ensemble of models for design - taking the minimum prediction across the
   target cell state and the maximum prediction across non-target cell states - or an average
   ensemble of models. We additionally generate predictions from a separate validation model
   and use these to generate plots.
4. Entry point: Parses command-line arguments and runs the main design function.

"""

import argparse
import datetime
import json
import os
import sys

import numpy
import matplotlib
from matplotlib import pyplot
import pandas
import seaborn

matplotlib.rcParams['savefig.dpi'] = 150
matplotlib.rcParams['savefig.bbox'] = 'tight'

import tensorflow
from tensorflow.keras import layers
import tensorflow_probability

import corefsp

BASE_DIR = '../'
sys.path.append(BASE_DIR)
import src.definitions
import src.model
import src.sequence
import src.plot
import src.utils

# Motif to penalize: GRHL
# pwm from TFModisco evl(50epi)/pwm_trimmed.meme: pos_patterns_pattern_1
PWM_TO_PENALIZE = [
    [0.260722, 0.336624, 0.174268, 0.228387,],
    [0.513274, 0.117086, 0.230088, 0.139551,],
    [0.715453, 0.021443, 0.143975, 0.119129,],
    [0.777740, 0.005106, 0.103131, 0.114023,],
    [0.001702, 0.994214, 0.003063, 0.001021,],
    [0.537100, 0.122532, 0.029612, 0.310756,],
    [0.138530, 0.010892, 0.755276, 0.095303,],
    [0.000681, 0.004765, 0.992852, 0.001702,],
    [0.091219, 0.031654, 0.007488, 0.869639,],
    [0.088155, 0.081688, 0.034377, 0.795779,],
    [0.154867, 0.144997, 0.102451, 0.597686,],
    [0.185500, 0.154867, 0.363513, 0.296120,],
]

# Paths for model files
DESIGN_MODEL_PATHS = [os.path.join(BASE_DIR, src.definitions.DEEPDANIO_MODEL_PATH[i]) for i in [1, 2]]
VAL_MODEL_PATH = os.path.join(BASE_DIR, src.definitions.DEEPDANIO_MODEL_PATH[0])

##################
# Loss functions #
##################
def get_target_mean_loss_func(
    target_idx,
    target_weight=1.0,
    non_target_weight=1.0,
    **kwargs,
):
    """
    Get a target loss function to provide to Fast SeqProp.
    
    The returned function maximizes the difference between a transformed target cell state
    prediction and the average of non-target cell state predictions.

    Parameters
    ----------
    target_idx : int
        Index of the target cell state to maximize.
    target_weight : float
        Weight for the target cell state score in the loss.
    non_target_weight : float
        Weight for the non-target cell state score in the loss.

    Returns
    -------
    function
        Loss function.

    """
    
    nontarget_cell_state_idx = [i for i in range(len(src.definitions.CELL_STATES)) if i!=target_idx]
    nontarget_cell_state_idx = tensorflow.cast(nontarget_cell_state_idx, tensorflow.int32)

    def target_loss_func(model_preds):
        # model_preds has dimensions (n_seqs, n_outputs)
        model_preds_target = model_preds[:, target_idx]
        model_preds_nontarget = tensorflow.gather(
            model_preds,
            nontarget_cell_state_idx,
            axis=1,
        )

        target_score_per_sample = tensorflow.where(
            model_preds_target < 0,
            model_preds_target,
            tensorflow.sqrt(model_preds_target),
        )
        target_score = - tensorflow.reduce_mean(target_score_per_sample)

        non_target_score = tensorflow.reduce_mean(model_preds_nontarget)

        return non_target_weight*non_target_score + target_weight*target_score

    return target_loss_func

def get_target_percentile_loss_func(
    target_idx,
    target_weight=1.0,
    non_target_weight=1.0,
    non_target_percentile=90,
    **kwargs,
):
    """
    Get a target loss function to provide to Fast SeqProp.
    
    The returned function maximizes the difference between a transformed target cell state
    prediction and a specified percentile of non-target cell state predictions.

    Parameters
    ----------
    target_idx : int
        Index of the target cell state to maximize.
    target_weight : float
        Weight for the target cell state score in the loss.
    non_target_weight : float
        Weight for the non-target cell state score in the loss.
    non_target_percentile : float
        Percentile of non-target cell state predictions to explicitly minimize.

    Returns
    -------
    function
        Loss function.

    """
    
    nontarget_cell_state_idx = [i for i in range(len(src.definitions.CELL_STATES)) if i!=target_idx]
    nontarget_cell_state_idx = tensorflow.cast(nontarget_cell_state_idx, tensorflow.int32)

    def target_loss_func(model_preds):
        # model_preds has dimensions (n_seqs, n_outputs)
        model_preds_target = model_preds[:, target_idx]
        model_preds_nontarget = tensorflow.gather(
            model_preds,
            nontarget_cell_state_idx,
            axis=1,
        )

        target_score_per_sample = tensorflow.where(
            model_preds_target < 0,
            model_preds_target,
            tensorflow.sqrt(model_preds_target),
        )
        target_score = - tensorflow.reduce_mean(target_score_per_sample)

        # non_target_score = tensorflow.reduce_mean(model_preds_nontarget)
        non_target_score = tensorflow.reduce_mean(
            tensorflow_probability.stats.percentile(
                model_preds_nontarget,
                non_target_percentile,
                interpolation='midpoint',
                axis=1,
            )
        )

        return non_target_weight*non_target_score + target_weight*target_score

    return target_loss_func

def get_repeat_loss_func():
    """
    Get a PWM loss function to provide to Fast SeqProp.

    The returned function penalizes repeated nucleotides in the PWM.

    Returns
    -------
    function
        Loss function.

    """
    def repeat_loss_func(pwm):
        # PWM has dimensions (n_seqs, seq_length, n_channels)
        return tensorflow.reduce_mean(pwm[:, :-1, :] * pwm[:, 1:, :])
    
    return repeat_loss_func

class PWMScore(layers.Layer):
    """
    Layer that computes a sliding similarity score to a given PWM.

    At every position, the maximum between the match score of the pwm or
    its reverse complement and the sequence is taken.

    """
    def __init__(self, pwm_val, score_threshold=None, **kwargs):
        super(PWMScore, self).__init__(**kwargs)
        # pwm_val is (pwm_len, 4)
        pwm_val = numpy.array(pwm_val)
        self.pwm_val = pwm_val
        # conv_filter should contain reverse complement
        conv_filter = numpy.array([pwm_val, pwm_val[::-1, ::-1]])
        # convert to (pwm_len, 4, 2)
        self.conv_filter = numpy.moveaxis(conv_filter, 0, -1)
        self.conv_filter = tensorflow.constant(self.conv_filter, dtype=tensorflow.float32)
        
        if score_threshold is None:
            self.score_threshold = 0.25*numpy.sum(pwm_val)
        else:
            self.score_threshold = score_threshold

    def call(self, inputs):
        # inputs is (batch_size, seq_length, 4)
        # output of conv1d is (batch_size, seq_length-k+1, 2)
        # output of this function is (batch_size, 4**k)
        conv_output = tensorflow.nn.conv1d(
            inputs,
            self.conv_filter,
            stride=1,
            padding='VALID',
        )
        # Get maximum across channel dimension
        max_output = tensorflow.reduce_max(conv_output, axis=2)
        
        relu_output = tensorflow.nn.relu(max_output - self.score_threshold)
        return relu_output
    
def get_pwm_match_loss_func(pwm_to_penalize, score_threshold=None):
    """
    Get a PWM match loss function to provide to Fast SeqProp.

    The returned function penalizes similarity to a given pwm.

    Parameters
    ----------
    pwm_to_penalize : numpy.ndarray
        PWM to penalize, of shape (pwm_length, 4).
    score_threshold : float, optional
        Score threshold. Defaults to 0.25 * sum of pwm_to_penalize.

    Returns
    -------
    function
        Loss function.
    
    """
    def pwm_match_loss_func(pwm):
        # pwm has dimensions (n_seqs, seq_length, n_channels)
        pwm_score_layer = PWMScore(pwm_to_penalize, score_threshold=score_threshold)
        return tensorflow.reduce_mean(pwm_score_layer(pwm))
    
    return pwm_match_loss_func


def get_pwm_repeat_and_match_loss_func(
        pwm_repeat_weight=1.0,
        pwm_match_weight=1.0,
        pwm_match_params={},
    ):
    """
    Get a combined PWM repeat and match loss function to provide to Fast SeqProp.
    """
    repeat_loss_func = get_repeat_loss_func()
    match_loss_func = get_pwm_match_loss_func(**pwm_match_params)

    def combined_loss_func(pwm):
        return pwm_repeat_weight * repeat_loss_func(pwm) + pwm_match_weight * match_loss_func(pwm)

    return combined_loss_func


#################################
# Main sequence design function #
#################################
def run(
        cell_state_idx,
        n_seqs,
        model_ensemble_type='pessimistic',
        target_loss_type='percentile',
        target_weight=1.0,
        non_target_weight=1.0,
        non_target_percentile=90,
        pwm_match_weight=1.0,
        output_dir='.',
        output_prefix=None,
        seed=None,
    ):
    """
    Run Fast SeqProp to generate sequences with cell state-specific activity using DeepDanio.

    Parameters
    ----------
    target_idx : int
        Index of the cell state to target within the available cell states.
    n_seqs : int
        Number of sequences to generate.
    model_ensemble_type : str, optional
        Type of model ensemble to use for design. Options are 'pessimistic' (default) or 'average'.
    target_loss_type : str, optional
        Type of target loss function to use. Options are 'percentile' (default) or 'mean'.
    target_weight : float, optional
        Weight for the target cell state score in the loss. Default is 1.0.
    non_target_weight : float, optional
        Weight for the non-target cell state score in the loss. Default is 1.0.
    non_target_percentile : float, optional
        Percentile of non-target cell state predictions to explicitly minimize. Default is 90.
    pwm_match_weight : float, optional
        Weight for PWM match penalty term in the PWM loss. Default is 1.0.
    output_dir : str, optional
        Directory to save output files. Default is current directory.
    output_prefix : str, optional
        Prefix for output files. If None, a prefix based on cell state index and name will be used.
    seed : int, optional
        Random seed for sequence initialization. If None, a random seed will be used.

    """

    os.makedirs(output_dir, exist_ok=True)

    cell_state = src.definitions.CELL_STATES[cell_state_idx]
    cell_state_sanitized = src.utils.sanitize_cell_state(cell_state)

    # Prefix for output files
    if output_prefix is None:
        output_prefix = f"{cell_state_idx}_{cell_state_sanitized}"

    print(f"Starting sequence design for cell state {cell_state} ({cell_state_idx + 1} / {len(src.definitions.CELL_STATES)})...")

    # Construct model for design
    # ==========================
    print("\nLoading design model...")

    # Load models to be used for design
    models_design_list = []
    for model_filepath in DESIGN_MODEL_PATHS:
        model = src.model.load_model(model_filepath)
        model._name = model_filepath.split('/')[-1].split('.')[0]
        models_design_list.append(model)

    # Create ensemble model
    if model_ensemble_type == 'pessimistic':
        # Pessimistic ensemble: minimum across target cell state, maximum across non-target cell states
        min_output_idx = [cell_state_idx]
        max_output_idx = [i for i in range(len(src.definitions.CELL_STATES)) if i != cell_state_idx]
        model_design = src.model.make_model_ensemble(
            models_design_list,
            min_output_idx=min_output_idx,
            max_output_idx=max_output_idx,
        )
    elif model_ensemble_type == 'average':
        # Average ensemble. Default is to average across all outputs.
        model_design = src.model.make_model_ensemble(
            models_design_list,
        )
    else:
        raise ValueError(f"Invalid model_ensemble_type: {model_ensemble_type}. Must be 'pessimistic' or 'average'.")

    # Generate sequences
    # ==================
    print("\nGenerating sequences...")

    # Run parameters
    run_parameters = {
        'run_id': datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        'target_idx': cell_state_idx,
        'fsp_params': {
            'seq_length': src.definitions.DEEPDANIO_INPUT_LENGTH,
            'n_seqs': n_seqs,
            'target_weight': 1,
            'pwm_weight': 1,
            'entropy_weight': 1e-3,
            'learning_rate': 0.001,
            'n_iter_max': 10000,
            'init_seed': seed,
            'early_stopping': True,
            'early_stopping_mode': 'rel',
            'early_stopping_min_delta': 0.001,
            'early_stopping_min_batch': 50,
        },
        'model_ensemble_type': model_ensemble_type,
        'target_loss_type': target_loss_type,
        'target_loss_params': {
            'target_weight': target_weight,
            'non_target_weight': non_target_weight,
            'non_target_percentile': non_target_percentile,
        },
        'pwm_loss_params': {
            'pwm_repeat_weight': 3.0,
            'pwm_match_weight': pwm_match_weight,
            'pwm_match_params': {
                'pwm_to_penalize': PWM_TO_PENALIZE,
            },
        },
    }

    # Get loss functions
    if target_loss_type == 'percentile':
        target_loss_func = get_target_percentile_loss_func(**run_parameters['target_loss_params'], target_idx=cell_state_idx)
    elif target_loss_type == 'mean':
        target_loss_func = get_target_mean_loss_func(**run_parameters['target_loss_params'], target_idx=cell_state_idx)
    else:
        raise ValueError(f"Invalid target_loss_type: {target_loss_type}. Must be 'percentile' or 'mean'.")
    pwm_loss_func = get_pwm_repeat_and_match_loss_func(**run_parameters['pwm_loss_params'])

    # Run Fast SeqProp
    with open(os.path.join(output_dir, f'{output_prefix}_run_metadata.json'), 'w') as file:
        file.write(json.dumps(run_parameters, indent=4))
    generated_onehot, generated_pred_design, train_history = corefsp.design_seqs(
        model_design,
        target_loss_func=target_loss_func,
        pwm_loss_func=pwm_loss_func,
        **run_parameters['fsp_params'],
    )

    # Save results
    # ============
    print("\nSaving results...")

    # Save sequences as fasta
    generated_seqs = src.sequence.one_hot_decode(generated_onehot)
    generated_seq_ids = [f'{output_prefix}_seq_{i}' for i in range(len(generated_seqs))]
    generated_df = pandas.DataFrame(
        {'seq_id': generated_seq_ids, 'seq': generated_seqs}
    )
    src.sequence.save_seqs_to_fasta(
        generated_df,
        os.path.join(output_dir, f"{output_prefix}_seqs.fasta"),
        id_col='seq_id',
        seq_col='seq',
    )

    # Save predictions from design model
    generated_design_preds_df = generated_df.copy()
    generated_design_preds_df[src.definitions.CELL_STATES] = generated_pred_design
    generated_design_preds_df.to_csv(
        os.path.join(output_dir, f"{output_prefix}_preds_design.csv.gz"),
        index=False,
    )
    
    # Generate predictions from validation model
    print("Generating predictions from validation model...")
    model_val = src.model.load_model(VAL_MODEL_PATH)
    generated_pred_val = model_val.predict(generated_onehot, verbose=1)
    
    generated_val_preds_df = generated_df.copy()
    generated_val_preds_df[src.definitions.CELL_STATES] = generated_pred_val
    generated_val_preds_df.to_csv(
        os.path.join(output_dir, f"{output_prefix}_preds_val.csv.gz"),
        index=False,
    )

    # Plots
    print("\nGenerating plots...")

    # Training history
    fig, axes = pyplot.subplots(1, len(train_history), figsize=(4*len(train_history), 3))
    for ax, (loss_component_key, loss_component_val) in zip(axes, train_history.items()):
        ax.plot(loss_component_val)
        ax.set_title(loss_component_key.replace('_', ' ').capitalize())
        ax.set_xlabel("Iteration")
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_train_history.png"))
    pyplot.close(fig)
    
    # Sequence bitmap
    ax = src.plot.plot_sequence_bitmap(generated_onehot)
    fig = ax.get_figure()
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_seq_bitmap.png"))
    pyplot.close(fig)

    # Sample sequences
    fig = src.plot.plot_seq_logos(generated_onehot, n_seqs=10)
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_seqs.png"))
    pyplot.close(fig)

    # Distribution of edit distances
    distances = src.sequence.get_paired_editdistances(generated_seqs)
    fig, ax = pyplot.subplots(1, 1, figsize=(4, 3.5))
    seaborn.violinplot(data=[distances], ax=ax)
    ax.set_xticks([])
    ax.set_ylim(0, 1)
    ax.set_ylabel(
        'Edit distance / nucleotide\n'
        '{:.3f} +/- {:.3f}'.format(numpy.mean(distances), numpy.std(distances))
    )
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_editdistance.png"))
    pyplot.close(fig)

    # Distribution of kmer distances
    try:
        distances = src.sequence.get_min_euc_nmer_dist(
            generated_seqs,
            nmer=4,
            random_seed_subsample=2020,
            normalize_counts=True,
        )
    except ValueError:
        distances = numpy.array([0]*len(generated_seqs))
    fig, ax = pyplot.subplots(1, 1, figsize=(4, 3.5))
    seaborn.violinplot(data=[distances], ax=ax)
    ax.set_xticks([])
    ax.set_ylim(0, 0.25)
    ax.set_ylabel(
        '4-mer distance\n'
        '{:.3f} +/- {:.3f}'.format(numpy.mean(distances), numpy.std(distances))
    )
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_4mer_distance.png"))
    pyplot.close(fig)

    # Design predictions box plot across cell states
    df_to_plot = generated_design_preds_df[src.definitions.CELL_STATES].melt(
        var_name='cell_state',
        value_name='prediction',
    )
    fig, ax = pyplot.subplots(figsize=(20, 3.5))
    seaborn.boxplot(
        data=df_to_plot,
        x='cell_state',
        y='prediction',
        order=src.definitions.CELL_STATES,
        fliersize=1,
        ax=ax,
    )
    rect = matplotlib.patches.Rectangle(
        (cell_state_idx - 0.5, ax.get_ylim()[0]),
        1, ax.get_ylim()[1] - ax.get_ylim()[0],
        linewidth=1.5,
        edgecolor='tab:red',
        facecolor='none',
        clip_on=False,
    )
    ax.add_patch(rect)
    ax.grid()
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    ax.set_xlabel('Cell state')
    ax.set_ylabel('$log_{10}$ accessibility prediction\nDesign model')
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_preds_design_boxplot.png"))
    pyplot.close(fig)

    # Validation predictions box plot across cell states
    df_to_plot = generated_val_preds_df[src.definitions.CELL_STATES].melt(
        var_name='cell_state',
        value_name='prediction',
    )
    fig, ax = pyplot.subplots(figsize=(20, 3.5))
    seaborn.boxplot(
        data=df_to_plot,
        x='cell_state',
        y='prediction',
        order=src.definitions.CELL_STATES,
        fliersize=1,
        ax=ax,
    )
    rect = matplotlib.patches.Rectangle(
        (cell_state_idx - 0.5, ax.get_ylim()[0]),
        1, ax.get_ylim()[1] - ax.get_ylim()[0],
        linewidth=1.5,
        edgecolor='tab:red',
        facecolor='none',
        clip_on=False,
    )
    ax.add_patch(rect)
    ax.grid()
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    ax.set_xlabel('Cell state')
    ax.set_ylabel('$log_{10}$ accessibility prediction\nValidation model')
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_preds_val_boxplot.png"))
    pyplot.close(fig)

    # Trajectory plot, design model predictions
    design_preds_median = generated_design_preds_df[src.definitions.CELL_STATES].median()
    vmin = design_preds_median.min()
    vmax = design_preds_median.max()
    fig = pyplot.figure(figsize=(8, 4), clear=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=1)
    ax_trajectory = fig.add_subplot(gs[0, 0], rasterized=True)

    # Plot trajectory
    src.plot.trajectory(
        design_preds_median,
        linewidth=1.25,
        markersize=20,
        cell_state_labels='final',
        cmap='viridis',
        ax=ax_trajectory,
    )

    # Add colorbar
    ax_colorbar = fig.add_subplot(gs[0, 1])
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = matplotlib.colorbar.ColorbarBase(
        ax_colorbar,
        cmap='viridis',
        norm=norm,
        orientation='vertical',
    )
    cbar.set_label('Observed $log_{10}$ accessibility', rotation=90, labelpad=10, fontsize=16)
    cbar.ax.tick_params(labelbottom=True, bottom=True, labelsize=16)

    fig.suptitle(f"Target: {cell_state.split('(')[0]}", y=0.95, fontsize=16)

    fig.savefig(os.path.join(output_dir, f"{output_prefix}_preds_design_trajectory.png"))
    pyplot.close(fig)

    # Trajectory plot, validation model predictions
    val_preds_median = generated_val_preds_df[src.definitions.CELL_STATES].median()
    vmin = val_preds_median.min()
    vmax = val_preds_median.max()
    fig = pyplot.figure(figsize=(8, 4), clear=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=1)
    ax_trajectory = fig.add_subplot(gs[0, 0], rasterized=True)

    # Plot trajectory
    src.plot.trajectory(
        val_preds_median,
        linewidth=1.25,
        markersize=20,
        cell_state_labels='final',
        cmap='viridis',
        ax=ax_trajectory,
    )

    # Add colorbar
    ax_colorbar = fig.add_subplot(gs[0, 1])
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = matplotlib.colorbar.ColorbarBase(
        ax_colorbar,
        cmap='viridis',
        norm=norm,
        orientation='vertical',
    )
    cbar.set_label('Observed $log_{10}$ accessibility', rotation=90, labelpad=10, fontsize=16)
    cbar.ax.tick_params(labelbottom=True, bottom=True, labelsize=16)

    fig.suptitle(f"Target: {cell_state.split('(')[0]}", y=0.95, fontsize=16)

    fig.savefig(os.path.join(output_dir, f"{output_prefix}_preds_val_trajectory.png"))
    pyplot.close(fig)

    print(f"\nDone with cell state {cell_state} ({cell_state_idx + 1} / {len(src.definitions.CELL_STATES)}).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Fast SeqProp to generate sequences specific to a DeepDanio-predicted cell state.')
    parser.add_argument('--target-idx', type=int, required=True, help='Target cell state index according to src/resources/cell_state_metadata.csv (0-based index)')
    parser.add_argument('--n-seqs', type=int, default=100, help='Number of sequences to design (default: 100)')
    parser.add_argument('--model-ensemble-type', type=str, choices=['pessimistic', 'average'], default='pessimistic', help='Model ensemble type (choices: pessimistic, average; default: pessimistic)')
    parser.add_argument('--target-loss-type', type=str, choices=['percentile', 'mean'], default='percentile', help='Target loss type (choices: percentile, mean; default: percentile)')
    parser.add_argument('--target-weight', type=float, default=1.0, help='Weight for maximization of target cell state prediction (default: 1.0)')
    parser.add_argument('--non-target-weight', type=float, default=1.0, help='Weight for minimization of non-target cell state predictions (default: 1.0)')
    parser.add_argument('--non-target-percentile', type=float, default=90, help='Percentile for non-target loss (default: 90)')
    parser.add_argument('--pwm-match-weight', type=float, default=1.0, help='Weight for PWM match penalty term (default: 1.0)')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory to store design results (default: results)')
    parser.add_argument('--output-prefix', type=str, default=None, help='Prefix for output files (default: auto-generated from target index and cell state)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for sequence initialization. (default: None, random initialization)')
    args = parser.parse_args()

    run(
        cell_state_idx=args.target_idx,
        n_seqs=args.n_seqs,
        target_weight=args.target_weight,
        non_target_weight=args.non_target_weight,
        non_target_percentile=args.non_target_percentile,
        pwm_match_weight=args.pwm_match_weight,
        model_ensemble_type=args.model_ensemble_type,
        target_loss_type=args.target_loss_type,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        seed=args.seed,
    )
