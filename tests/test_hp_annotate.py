from pathlib import Path
import string

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ps_funks import hotPlot as hp


def test_annotate_creates_png(tmp_path):
    rng = np.random.default_rng(42)
    n_points = 30
    x = rng.normal(size=n_points)
    y = rng.normal(size=n_points)

    alphabet = np.array(list(string.ascii_letters))
    lengths = rng.integers(3, 11, size=n_points)
    labels = [
        "".join(rng.choice(alphabet, size=int(length), replace=True))
        for length in lengths
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=30)
    hp.annotate(ax, x, y, labels, fontsize=8)

    out = Path(tmp_path) / "annotate_random_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    assert out.exists()
    assert out.stat().st_size > 0
