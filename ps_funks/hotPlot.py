import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from cycler import cycler
import numpy as np
import ps_funks.juteUtils as jut
from pathlib import Path
import seaborn as sns


def setRcParams(cycleLinestyles=False, cycleColors=True, spinesRight=False):
    if cycleColors:
        matplotlib.rcParams["axes.prop_cycle"] = cycler("color", cssCblind)
    if cycleLinestyles:
        # color AND linestyle cycle for lines
        # However, cycler + cycler of different length shortens the longer one
        matplotlib.rcParams["axes.prop_cycle"] += cycler("linestyle", lss)
    # make that only left and bottom spines are shown
    matplotlib.rcParams["axes.spines.top"] = False
    matplotlib.rcParams["axes.spines.right"] = spinesRight
    matplotlib.rcParams["font.family"] = "sans-serif"
    # facecolors
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["savefig.facecolor"] = "white"


def savefig_multiformat(dir, f_name, f, formats=["png", "pdf", "svg"]):
    """
    THE IDEA at the start of notebook partialize the function by:
        savefig_multi = partial(savefig_multiformat, dir=dir)

    INPUT:
        dir Path
            directory to save the figure
        f_name str
            name of the figure
        f matplotlib.figure.Figure
            figure to save
        formats list
            list of formats to save the figure
    """
    dir = Path(dir)
    dir.mkdir(exist_ok=True)
    for fmt in formats:
        dir_ = dir / fmt
        dir_.mkdir(exist_ok=True)
        f.tight_layout()
        f.savefig(dir_ / (f_name + "." + fmt), dpi=300, bbox_inches="tight", pad_inches=0)


def axesGrid(N, size=0.8, aspect=1, flatten=True, n_col=None, n_row=None, **kwgs):
    """
    returns
    INPUT:
        aspect float
            changes the aspect of each axis
            e.g.: 0.5: |__
                  2: |
                     |_
        flatten bool
            Default=True
            True: returns axs.flatten()[:N]
            False: returns axs-array
    """
    m, n = rows_and_cols(N, n_row, n_col)
    f, axs = plt.subplots(m, n, figsize=m * size * plt.figaspect(m / n * aspect), **kwgs)
    if m * n > N:
        axs = axs.flatten()
        _ = [ax.remove() for ax in axs[N:]]
        axs = axs.reshape(m, n)
    if flatten and N > 1:
        axs = axs.flatten()[:N]
    return f, axs


def rows_and_cols(N, n_row, n_col):
    if n_row is not None:
        m = n_row
        n = int((t := N / m)) + int(np.mod(t, 1) > 0)
    elif n_col is not None:
        n = n_col
        m = int((t := N / n)) + int(np.mod(t, 1) > 0)
    else:
        m = int(np.sqrt(N))
        n = m + int((m - np.sqrt(N)) < 0)
        m += int((n * m - N) < 0)
    return m, n


def abc_plotLabels(
    coord,
    axs,
    lower_case=False,
    fontsize=22,
    Nskip=0,
    abc=None,
    facecolors=None,
    edgecolors=None,
    **kwgs,
):
    """
    INPUT:
        coord.shape (2)
            coordinates relative to borders of plot
        axs [matplotlib.subplotobject, ....]
            list of subplots which needs label
    """
    abc = jut.setDefault(abc, get_ABC(len(axs), Nskip=Nskip, lower_case=lower_case))
    for i, ax in enumerate(axs):
        dic = {}
        if facecolors is not None:
            dic["facecolor"] = facecolors[i]
        if edgecolors is not None:
            dic["edgecolor"] = edgecolors[i]
        if edgecolors is None and facecolors is None:
            dic = None
        ax.text(
            coord[0],
            coord[1],
            abc[i],
            fontsize=fontsize,
            transform=ax.transAxes,
            bbox=dic,
            **kwgs,
        )


def get_ABC(N, Nskip=None, lower_case=False):
    Nskip = jut.setDefault(Nskip, 0)
    # fmt: off
    abc = [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    ]
    ABC = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    ]
    # fmt: on
    if not lower_case:
        abc = ABC
    return abc[Nskip : Nskip + N]


def rotate_xticklabels(ax, deg):
    for tick in ax.get_xticklabels():
        tick.set_rotation(deg)


def rotate_yticklabels(ax, deg):
    for tick in ax.get_yticklabels():
        tick.set_rotation(deg)


def hist_logx(ax, x, bins, **kwargs):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    logbins[-1] += logbins[0] / 100  # to include the last datapoint
    ax.hist(x, bins=logbins, **kwargs)
    ax.set(xscale="log")


def nice_dates(ax, y=False, monthstep=2, month_offset=1, rotation_major=0, rotation_minor=None):
    assert isinstance(monthstep, int), "monthstep must be integer"
    assert isinstance(month_offset, int), "month_offset must be integer"
    axis = ax.yaxis if y else ax.xaxis
    axis.set_minor_locator(matplotlib.dates.MonthLocator(np.arange(1, 13)))
    axis.set_minor_formatter(DateFormatter_withEmptyStrings("%b", monthstep))
    axis.set_major_locator(matplotlib.dates.YearLocator())
    axis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    if rotation_major is not None:
        for t in axis.get_majorticklabels():
            t.set_rotation(rotation_major)
            t.set_horizontalalignment("center")
    if rotation_minor is not None:
        for t in axis.get_minorticklabels():
            t.set_rotation(rotation_minor)


class DateFormatter_withEmptyStrings(matplotlib.dates.DateFormatter):
    # def __init__(self, fmt, month_step, **kwargs):
    def __init__(self, fmt, month_step, tz=None):
        super().__init__(fmt, tz=tz, usetex=False)
        self._month_step = month_step

    def __call__(self, x, pos=0):
        out = super().__call__(x, pos)
        if self._month_step > 1:
            months_short = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            keep = list(
                set(months_short[:: self._month_step])
                - set(months_short[(-self._month_step + 1) :])
            )
            out = out if out in keep else ""
        return out


def text_only_plot(ax, txt, **kwgs):
    # in order to have well-aligned text on top, the code should start there
    ax.text(0, 1, txt, va="top", **kwgs)
    ax.axis("off")
    ax.set(title="")


def text_input_output_plot(txt_input, txt_output, wspace=None, from_file=True, max_length=None):
    """
    INPUT:
        txt_input str OR txt-file
        txt_output str OR txt-file
        wspace float
            horizontal (width) space between the subplots
            should be set to larger values if text txt_input is long

    NOTE:
        remember to save with option "bbox_inches='tight'"
            f, axs = text_input_output_plot(in, out, wspace=1)
            f.savefig('test.pdf', bbox_inches='thight')
    """
    if from_file:
        txt_input = Path(txt_input).open("r").read()
        txt_output = Path(txt_output).open("r").read()
    if max_length is not None:
        txt_input = jut.text_addLineBreaks(txt_input, max_length)
        txt_input = jut.text_addLineBreaks(txt_input, max_length, breaker="-")
        txt_input = jut.text_addLineBreaks(txt_input, max_length, breaker=".")
        txt_output = jut.text_addLineBreaks(txt_output, max_length)
        txt_output = jut.text_addLineBreaks(txt_output, max_length, breaker="-")
        txt_output = jut.text_addLineBreaks(txt_output, max_length, breaker=".")
    gridspec_kw = {}
    if wspace is not None:
        gridspec_kw["wspace"] = wspace
    f, axs = axesGrid(2, gridspec_kw=gridspec_kw)
    _ = [text_only_plot(ax, txt) for ax, txt in zip(axs, [txt_input, txt_output])]
    _ = [ax.set_title(ti, fontsize=12, weight="bold") for ax, ti in zip(axs, ["input:", "output:"])]
    return f, axs


def swarmplot_single(ax, df, col, orient="h", **kwgs):
    """
    In order for sns.swarmplot to use "hue" coloring, an y-value must be given
        * create a dummy y value
    """

    df[""] = ""
    if orient == "h":
        x, y = col, ""
    else:
        y, x = col, ""
    sns.swarmplot(data=df, x=x, y=y, ax=ax, orient=orient, **kwgs)
    undo_seaborn_xaxis(ax)
    if orient == "h":
        yAxisOff(ax)
    else:
        xAxisOff(ax)
    df.drop(columns=[""], inplace=True)


def yAxisOff(axs):
    axs.get_yaxis().set_visible(False)  # no ticks
    axs.spines["left"].set_visible(False)  # no spine


def xAxisOff(axs):
    axs.get_xaxis().set_visible(False)  # no ticks
    axs.spines["bottom"].set_visible(False)  # no spine


def undo_seaborn_xaxis(ax):
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("k")
    ax.tick_params(axis="x", which="both", reset=True, top=False)


def legend_sorted(ax, reverse=False, **kwgs):
    return sorted_legend(ax, reverse=reverse, **kwgs)


def sorted_legend(ax, reverse=False, **kwgs):
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    if reverse:
        labels, handles = labels[::-1], handles[::-1]
    return ax.legend(handles, labels, **kwgs)


def legend_handle_same_size(legend, size=30):
    for handle in legend.legendHandles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([size])
        elif hasattr(handle, "set_linewidth"):
            handle.set_linewidth(size)
        else:
            print("legend handle is neither marker nor line")
            print(f"its: {type(handle)}")


def undo_seaborn_params():
    matplotlib.rcParams["xtick.bottom"] = True
    matplotlib.rcParams["ytick.left"] = True
    matplotlib.rcParams["axes.grid"] = False
    matplotlib.rcParams["axes.spines.top"] = False
    matplotlib.rcParams["axes.spines.right"] = False
    matplotlib.rcParams["axes.spines.bottom"] = True
    matplotlib.rcParams["axes.spines.left"] = True
    matplotlib.rcParams["axes.edgecolor"] = "k"


def color_to_rgb(c):
    return mplc.to_rgb(c)


# extended list of linestyles, linestyletuples are created via (offset,(xpt line, xpt space, line, space, ...))
lss = [
    "-",
    ":",
    "--",
    "-.",
    (0, (1, 1, 1, 4)),  # = .. .. (double dots)
    (0, (5, 1, 1, 1, 1, 1)),  # = -..-.. (dash double dots)
    (0, (7, 2)),  # = -- -- (long dashes)
    (0, (1, 1, 1, 1, 1, 4)),  # = ... ... (triple dots)
    (0, (1, 1, 1, 1, 1, 1, 6, 1)),
]  # = ...-...- (triple dots dash)

# colorpalette optimized for colorblind people: https://www.nature.com/articles/nmeth.1618
# * other noce websites:
#       * https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40
#       * https://gka.github.io/palettes/#/9%7Cd%7C0033a6,0098ce,ecf8fa%7Cffb5a1,ff5659,b40000%7C1%7C1
cssCblind_A = [
    (0, 0, 0),
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (0, 114, 178),
    (213, 94, 0),
    (204, 121, 167),
]
cssCblind = [(t[0] / 255, t[1] / 255, t[2] / 255) for t in cssCblind_A]
