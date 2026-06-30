from pathlib import Path

import numpy
import numpy as np
import matplotlib
from matplotlib import pyplot
import pandas

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import CubicSpline

import logomaker

import src.definitions

# Nucleotide colors
NT_COLOR_DICT = {
    'A': (15/255, 148/255, 71/255),
    'C': (35/255, 63/255, 153/255),
    'G': (245/255, 179/255, 40/255),
    'T': (228/255, 38/255, 56/255),
}

CELL_STATE_COORDS = {
    'high cells': (0, 0),
    
    'deep cells(oblong)': (1, 0),
    'evl(oblong)': (1, -13),

    'ectoderm(dome)': (2, 2),
    'dorsal margin(dome)': (2, 1),
    'non-dorsal margin(dome)': (2, 0),
    'apoptosis like(dome)': (2, -1),
    'ysl(dome)': (2, -12),
    'evl(dome)': (2, -13),

    'ectoderm(30epi)': (3, 2),
    'dorsal margin(30epi)': (3, 1),
    'non-dorsal margin(30epi)': (3, 0),
    'apoptosis like(30epi)': (3, -1),
    'ysl(30epi)': (3, -12),
    'evl(30epi)': (3, -13),

    'dorsal ectoderm(50epi)': (4, 3),
    'ventral ectoderm(50epi)': (4, 2),
    'dorsal anterior(50epi)': (4, 1),
    'dorsal posterior(50epi)': (4, 0),
    'margin tail(50epi)': (4, -1),
    'ventrolateral mesoendoderm(50epi)': (4, -2),
    'apoptosis like(50epi)': (4, -3),
    'ysl(50epi)': (4, -12),
    'evl(50epi)': (4, -13),

    'dorsal ectoderm(shield)': (5, 3),
    'ventral ectoderm(shield)': (5, 2),
    'dorsal anterior(shield)': (5, 1),
    'dorsal posterior(shield)': (5, 0),
    'margin tail(shield)': (5, -1),
    'ventrolateral mesoderm(shield)': (5, -2),
    'endoderm(shield)': (5, -3),
    'ysl(shield)': (5, -12),
    'evl(shield)': (5, -13),

    'neural plate anterior(75epi)': (6, 4),
    'neural plate posterior(75epi)': (6, 3),
    'epidermis(75epi)': (6, 2),
    'prechordal plate(75epi)': (6, 1),
    'notochord(75epi)': (6, 0),
    'margin tail(75epi)': (6, -1),
    'lateral plate mesoderm(75epi)': (6, -2),
    'psm(75epi)': (6, -3),
    'adaxial cells(75epi)': (6, -4),
    'endoderm(75epi)': (6, -5),
    'forerunner cells(75epi)': (6, -6),
    'ysl(75epi)': (6, -12),
    'evl(75epi)': (6, -13),

    'telencephalon(bud)': (7, 9),
    'dorsal diencephalon(bud)': (7, 8),
    'optic primordium(bud)': (7, 7),
    'ventral diencephalon(bud)': (7, 6),
    'midbrain(bud)': (7, 5),
    'hindbrain(bud)': (7, 4),
    'spinal cord(bud)': (7, 3),
    'anterior neural plate border(bud)': (7, 2),
    'epidermis(bud)': (7, 1),
    'posterior neural plate border(bud)': (7, 0),
    'hatching gland(bud)': (7, -1),
    'notochord(bud)': (7, -2),
    'tailbud(bud)': (7, -3),
    'lateral plate mesoderm(bud)': (7, -4),
    'cephalic mesoderm(bud)': (7, -5),
    'psm(bud)': (7, -6),
    'adaxial cells(bud)': (7, -7),
    'endoderm(bud)': (7, -8),
    'ysl(bud)': (7, -12),
    'evl(bud)': (7, -13),

    'telencephalon(6somite)': (8, 15),
    'dorsal diencephalon(6somite)': (8, 14),
    'optic primordium(6somite)': (8, 13),
    'ventral diencephalon(6somite)': (8, 12),
    'neural floorplate(6somite)': (8, 11),
    'midbrain(6somite)': (8, 10),
    'hindbrain(6somite)': (8, 9),
    'spinal cord(6somite)': (8, 8),
    'differentiating neuron(6somite)': (8, 7),
    'neural crest(6somite)': (8, 6),
    'olfactory/adenohypophyseal placode(6somite)': (8, 5),
    'lens placode(6somite)': (8, 4),
    'otic placode(6somite)': (8, 3),
    'ionocyte progenitors(6somite)': (8, 2),
    'epidermis(6somite)': (8, 1),
    'posterior neural plate border(6somite)': (8, 0),
    'hatching gland(6somite)': (8, -1),
    'notochord(6somite)': (8, -2),
    'tailbud(6somite)': (8, -3),
    'endothelial progenitors(6somite)': (8, -4),
    'heart field(6somite)': (8, -5),
    'pronephric duct+blood island(6somite)': (8, -6),
    'cephalic mesoderm(6somite)': (8, -7),
    'psm(6somite)': (8, -8),
    'somite(6somite)': (8, -9),
    'adaxial cells(6somite)': (8, -10),
    'endoderm(6somite)': (8, -11),
    'ysl(6somite)': (8, -12),
    'evl(6somite)': (8, -13),
}

COLOR_NEURAL_ECTODERM = 'darkgreen'
COLOR_NONNEURAL_ECTODERM = 'darkred'
COLOR_MESODERM = 'darkgoldenrod'
COLOR_ENDODERM = 'tab:blue'
COLOR_YSL = 'tab:pink'
COLOR_EVL = 'tab:orange'
COLOR_OTHER = 'k'

CELL_STATE_DEFAULT_COLORS = {
    'high cells': COLOR_OTHER,
    
    'deep cells(oblong)': COLOR_OTHER,
    'evl(oblong)': COLOR_EVL,

    'ectoderm(dome)': COLOR_OTHER,
    'dorsal margin(dome)': COLOR_MESODERM,
    'non-dorsal margin(dome)': COLOR_OTHER,
    'apoptosis like(dome)': COLOR_OTHER,
    'ysl(dome)': COLOR_YSL,
    'evl(dome)': COLOR_EVL,

    'ectoderm(30epi)': COLOR_OTHER,
    'dorsal margin(30epi)': COLOR_MESODERM,
    'non-dorsal margin(30epi)': COLOR_OTHER,
    'apoptosis like(30epi)': COLOR_OTHER,
    'ysl(30epi)': COLOR_YSL,
    'evl(30epi)': COLOR_EVL,

    'dorsal ectoderm(50epi)': COLOR_NEURAL_ECTODERM,
    'ventral ectoderm(50epi)': COLOR_NONNEURAL_ECTODERM,
    'dorsal anterior(50epi)': COLOR_MESODERM,
    'dorsal posterior(50epi)': COLOR_MESODERM,
    'margin tail(50epi)': COLOR_MESODERM,
    'ventrolateral mesoendoderm(50epi)': COLOR_OTHER,
    'apoptosis like(50epi)': COLOR_OTHER,
    'ysl(50epi)': COLOR_YSL,
    'evl(50epi)': COLOR_EVL,

    'dorsal ectoderm(shield)': COLOR_NEURAL_ECTODERM,
    'ventral ectoderm(shield)': COLOR_NONNEURAL_ECTODERM,
    'dorsal anterior(shield)': COLOR_MESODERM,
    'dorsal posterior(shield)': COLOR_MESODERM,
    'margin tail(shield)': COLOR_MESODERM,
    'ventrolateral mesoderm(shield)': COLOR_MESODERM,
    'endoderm(shield)': COLOR_ENDODERM,
    'ysl(shield)': COLOR_YSL,
    'evl(shield)': COLOR_EVL,

    'neural plate anterior(75epi)': COLOR_NEURAL_ECTODERM,
    'neural plate posterior(75epi)': COLOR_NEURAL_ECTODERM,
    'epidermis(75epi)': COLOR_NONNEURAL_ECTODERM,
    'prechordal plate(75epi)': COLOR_MESODERM,
    'notochord(75epi)': COLOR_MESODERM,
    'margin tail(75epi)': COLOR_MESODERM,
    'lateral plate mesoderm(75epi)': COLOR_MESODERM,
    'psm(75epi)': COLOR_MESODERM,
    'adaxial cells(75epi)': COLOR_MESODERM,
    'endoderm(75epi)': COLOR_ENDODERM,
    'forerunner cells(75epi)': COLOR_ENDODERM,
    'ysl(75epi)': COLOR_YSL,
    'evl(75epi)': COLOR_EVL,

    'telencephalon(bud)': COLOR_NEURAL_ECTODERM,
    'dorsal diencephalon(bud)': COLOR_NEURAL_ECTODERM,
    'optic primordium(bud)': COLOR_NEURAL_ECTODERM,
    'ventral diencephalon(bud)': COLOR_NEURAL_ECTODERM,
    'midbrain(bud)': COLOR_NEURAL_ECTODERM,
    'hindbrain(bud)': COLOR_NEURAL_ECTODERM,
    'spinal cord(bud)': COLOR_NEURAL_ECTODERM,
    'anterior neural plate border(bud)': COLOR_NONNEURAL_ECTODERM,
    'epidermis(bud)': COLOR_NONNEURAL_ECTODERM,
    'posterior neural plate border(bud)': COLOR_NONNEURAL_ECTODERM,
    'hatching gland(bud)': COLOR_MESODERM,
    'notochord(bud)': COLOR_MESODERM,
    'tailbud(bud)': COLOR_MESODERM,
    'lateral plate mesoderm(bud)': COLOR_MESODERM,
    'cephalic mesoderm(bud)': COLOR_MESODERM,
    'psm(bud)': COLOR_MESODERM,
    'adaxial cells(bud)': COLOR_MESODERM,
    'endoderm(bud)': COLOR_ENDODERM,
    'ysl(bud)': COLOR_YSL,
    'evl(bud)': COLOR_EVL,

    'telencephalon(6somite)': COLOR_NEURAL_ECTODERM,
    'dorsal diencephalon(6somite)': COLOR_NEURAL_ECTODERM,
    'optic primordium(6somite)': COLOR_NEURAL_ECTODERM,
    'ventral diencephalon(6somite)': COLOR_NEURAL_ECTODERM,
    'neural floorplate(6somite)': COLOR_NEURAL_ECTODERM,
    'midbrain(6somite)': COLOR_NEURAL_ECTODERM,
    'hindbrain(6somite)': COLOR_NEURAL_ECTODERM,
    'spinal cord(6somite)': COLOR_NEURAL_ECTODERM,
    'differentiating neuron(6somite)': COLOR_NEURAL_ECTODERM,
    'neural crest(6somite)': COLOR_OTHER,
    'olfactory/adenohypophyseal placode(6somite)': COLOR_NONNEURAL_ECTODERM,
    'lens placode(6somite)': COLOR_NONNEURAL_ECTODERM,
    'otic placode(6somite)': COLOR_NONNEURAL_ECTODERM,
    'ionocyte progenitors(6somite)': COLOR_NONNEURAL_ECTODERM,
    'epidermis(6somite)': COLOR_NONNEURAL_ECTODERM,
    'posterior neural plate border(6somite)': COLOR_NONNEURAL_ECTODERM,
    'hatching gland(6somite)': COLOR_MESODERM,
    'notochord(6somite)': COLOR_MESODERM,
    'tailbud(6somite)': COLOR_MESODERM,
    'endothelial progenitors(6somite)': COLOR_MESODERM,
    'heart field(6somite)': COLOR_MESODERM,
    'pronephric duct+blood island(6somite)': COLOR_MESODERM,
    'cephalic mesoderm(6somite)': COLOR_MESODERM,
    'psm(6somite)': COLOR_MESODERM,
    'somite(6somite)': COLOR_MESODERM,
    'adaxial cells(6somite)': COLOR_MESODERM,
    'endoderm(6somite)': COLOR_ENDODERM,
    'ysl(6somite)': COLOR_YSL,
    'evl(6somite)': COLOR_EVL,
}

def plot_sequence_bitmap(seq_vals, ax=None, legend=True):
    """
    Plot a list of sequences as a colormap, with each sequence in its own row.

    Parameters
    ----------
    seq_vals : list of str or numpy.ndarray
        List of sequences to plot. Could be raw strings or one hot-encoded.
    ax : matplotlib.Axes or None
        Axes to plot on. If None, create a new figure.
    legend : bool
        Whether to include a legend for nucleotide colors.

    Returns
    -------
    matplotlib.Axes
        The Axes object with the plot.

    """
    # Convert sequences to numerical indices
    if type(seq_vals[0])==numpy.ndarray:
        # Assume one hot-encoded
        seqs_as_index = numpy.argmax(seq_vals, axis=-1)
    elif type(seq_vals[0])==str:
        # Assume list of strings
        seqs_as_index = [[["A", "C", "G", "T"].index(c) for c in si] for si in seq_vals]
        seqs_as_index = numpy.array(seqs_as_index)
    else:
        raise ValueError(f"type of seq_vals {type(seq_vals)} not recognized")
            
    # Define colors and colormap
    nt_colors = [NT_COLOR_DICT[n] for n in ['A', 'C', 'G', 'T']]
    cmap = matplotlib.colors.ListedColormap(nt_colors)
    bounds=[0, 1, 2, 3, 4]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Actually plot
    if ax is None:
        fig, ax = pyplot.subplots(figsize=(0.03*seqs_as_index.shape[1], 0.03*seqs_as_index.shape[0]))
    else:
        fig = ax.figure
    ax.imshow(
        seqs_as_index[::-1] + 0.5,
        aspect='equal',
        interpolation='nearest',
        origin='lower',
        cmap=cmap,
        norm=norm,
    )
    # Custom legend with nucleotide colors
    if legend:
        legend_elements = [
            matplotlib.patches.Patch(facecolor=c, label=nt)
            for nt, c in NT_COLOR_DICT.items()
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1.015), loc='upper left', fontsize='medium')

    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_xlabel('Position (nt)')
    ax.set_ylabel('Generated sequences')

    return ax

def plot_seq_logo(nt_height=None, pwm=None, seq=None, font_name='DejaVu Sans Mono', ax=None, title=None):
    """
    Plot a sequence logo
    
    Input can be specified as a sequence string, pwm matrix, or nucleotide height matrix.

    Parameters
    ----------
    nt_height : numpy.ndarray or None
        Nucleotide height matrix.
    pwm : numpy.ndarray or None
        PWM matrix.
    seq : str or None
        Sequence string.
    font_name : str
        Font name for the logo text.
    ax : matplotlib.Axes or None
        Axes to plot on.
    title : str or None
        Title for the plot.

    Returns
    -------
    matplotlib.Axes
        Axes containing the sequence logo.

    """

    if nt_height is None and seq is None and pwm is None:
        raise ValueError("At least one of nt_height, seq, or pwm must be provided")
    
    # Preference is given to nt_height, then pwm, then seq
    if nt_height is None:
        if pwm is not None:
            # Infer nucleotide heights using information content / entropy
            entropy = numpy.zeros_like(pwm)
            entropy[pwm > 0] = pwm[pwm > 0] * -numpy.log2(pwm[pwm > 0])
            entropy = numpy.sum(entropy, axis=1)
            conservation = 2 - entropy
            # Nucleotide height
            nt_height = numpy.tile(numpy.reshape(conservation, (-1, 1)), (1, 4))
            nt_height = pwm * nt_height
        elif seq is not None:
            # Nucleotide heights from one hot-encoding of sequence
            nt_to_onehot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
            nt_height = [nt_to_onehot[c] for c in seq.upper()]
            nt_height = numpy.array(nt_height)

    nt_height_df = pandas.DataFrame(
        nt_height,
        columns=['A', 'C', 'G', 'T'],
    )

    if ax is None:
        fig, ax = pyplot.subplots(figsize=(len(nt_height_df)/20, 0.5))
    
    logo = logomaker.Logo(
        nt_height_df,
        color_scheme=NT_COLOR_DICT,
        ax=ax,
        font_name=font_name,
    )
    logo.style_spines(visible=False)
    logo.style_spines(spines=['bottom'], visible=True, linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)

    return ax

def plot_seq_logos(seq_vals, n_seqs=None):
    """
    Plot sequence logos for a list of sequences.

    Parameters
    ----------
    seq_vals : list
        List of sequence strings or nucleotide height matrices.
    n_seqs : int, optional
        Number of sequences to plot. If None, plot all sequences.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the sequence logos.

    """
    if n_seqs is None:
        n_seqs = len(seq_vals)
    else:
        n_seqs = min(len(seq_vals), n_seqs)
    seq_len = len(seq_vals[0])

    fig, axes = pyplot.subplots(n_seqs, 1, figsize=(seq_len/10, 0.4*n_seqs))
    for seq_idx in range(n_seqs):
        if isinstance(seq_vals[seq_idx], str):
            plot_seq_logo(seq=seq_vals[seq_idx], ax=axes[seq_idx])
        else:
            plot_seq_logo(nt_height=seq_vals[seq_idx], ax=axes[seq_idx])

    return fig


def trajectory(
        cell_state_values=None,
        cmap=None,
        cell_state_colors=None,
        shade_edges=True,
        cell_state_labels=None,
        markersize=25,
        linewidth=1,
        cell_state_label_size=8,
        vmin=None,
        vmax=None,
        figsize=(8, 6),
        ax=None,
    ):
    """
    Plot the cell state differentiation trajectory with an overlaid colormap.

    Parameters
    ----------
    cell_state_values : dict or pandas.Series, optional
        Dictionary or Series mapping cell states to values for coloring.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use for coloring cell states based on values.
    cell_state_colors : dict, optional
        Dictionary mapping cell states to specific colors. If provided, this will
        override `cell_state_values` and `cmap`.
    shade_edges : bool, optional
        Whether to shade edges between cell states based on their colors.
    cell_state_labels : str, optional
        If 'all', label all cell states. If 'final', label only final cell states.
    markersize : int, optional
        Size of the markers for cell states.
    linewidth : int, optional
        Width of the lines for edges between cell states.
    cell_state_label_size : int, optional
        Font size for cell state labels.
    vmin : float, optional
        Minimum value for normalizing `cell_state_values` when using a colormap.
    vmax : float, optional
        Maximum value for normalizing `cell_state_values` when using a colormap.
    figsize : tuple, optional
        Size of the figure.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes will be created.

    """

    cell_states = src.definitions.CELL_STATES

    # Load edge probabilities table from package resources by default.
    edge_prob_path = Path(__file__).resolve().parent / 'resources' / 'edge_prob.txt'
    edge_prob_df = pandas.read_csv(edge_prob_path, sep='\t', header=None)
    edge_prob_df.columns = ['cell_state_1', 'cell_state_2', 'edge_prob']

    # Decide on cell state colors
    if cell_state_values is not None:
        if cmap is None:
            cmap = matplotlib.cm.get_cmap('viridis')
        elif type(cmap) == str:
            cmap = matplotlib.cm.get_cmap(cmap)
        
        # Normalize values
        if type(cell_state_values) == dict:
            cell_state_values = numpy.array([cell_state_values[cell_state] for cell_state in cell_states])
        elif type(cell_state_values) == pandas.Series:
            cell_state_values = numpy.array(cell_state_values[cell_states].astype(float))
        cell_state_values = numpy.array(cell_state_values)
        if vmin is None:
            vmin = numpy.min(cell_state_values)
        if vmax is None:
            vmax = numpy.max(cell_state_values)
        cell_state_values = (cell_state_values - vmin) / (vmax - vmin)

        # Get colors from colormap
        cell_state_colors = cmap(cell_state_values)
        cell_state_colors = {cell_state: color for cell_state, color in zip(cell_states, cell_state_colors)}

    elif cell_state_colors is None:
        cell_state_colors = CELL_STATE_DEFAULT_COLORS.copy()

    if ax is None:
        fig, ax = pyplot.subplots(figsize=figsize)

    # Plot one dot for each cell state
    for cell_state, coords_ in CELL_STATE_COORDS.items():
        x, y = coords_
        ax.scatter(x, y, color=cell_state_colors[cell_state], s=markersize, zorder=2)
        if cell_state_labels == 'all':
            ax.text(x, y, cell_state.split('(')[0], ha='left', va='bottom', fontsize=cell_state_label_size, zorder=3, rotation=45)
        elif cell_state_labels == 'final' and '(6somite)' in cell_state:
            ax.text(x + 0.2, y, cell_state.split('(')[0], ha='left', va='center', fontsize=cell_state_label_size, zorder=3)

    # Plot edges
    for i, row in edge_prob_df.iterrows():
        cell_state_1 = row['cell_state_1']
        cell_state_2 = row['cell_state_2']
        edge_prob = row['edge_prob']

        x1, y1 = CELL_STATE_COORDS[cell_state_1]
        x2, y2 = CELL_STATE_COORDS[cell_state_2]

        # Draw line between the two cell states
        # Line should be a spline with zero slopes on both ends
        x_line = numpy.linspace(x1, x2, 100)
        y_line = CubicSpline([x1, x2], [y1, y2], bc_type='clamped')(x_line)

        if shade_edges:
            # Make custom colormap from cell state 1 color to cell state 2 color
            colors = [cell_state_colors[cell_state_1], cell_state_colors[cell_state_2]]
            cm = LinearSegmentedColormap.from_list("Custom", colors, N=100)
            # Create line segments so we can color them individually
            points = numpy.array([x_line, y_line]).T.reshape(-1, 1, 2)
            segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
            norm = pyplot.Normalize(0, 1)
            lc = LineCollection(
                segments,
                cmap=cm,
                norm=norm,
                array=numpy.linspace(0, 1, len(segments)),
                linewidth=linewidth,
                alpha=edge_prob,
                )
            line = ax.add_collection(lc)
        else:
            ax.plot(x_line, y_line, color='black', alpha=edge_prob, lw=1, zorder=1)

    ax.set_xlim(-0.5, 8.5)

    # Disable axes borders, ticks, and tickmarks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    return ax
