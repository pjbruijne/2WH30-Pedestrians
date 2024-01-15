from cycler import cycler
import matplotlib as mpl


COLORS = ["#003F5C", "#FF6361", "#58508D", "#FFA600"]
LINE_CYCLER = cycler(color=COLORS) + cycler(linestyle=["-", "--", ":", "-"])

mpl.rcParams.update(
    {
        "font.size": 14,
        "legend.fontsize": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "lines.linewidth": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.prop_cycle": LINE_CYCLER,
    }
)
