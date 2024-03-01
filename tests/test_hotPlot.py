from ps_funks import hotPlot as hp
import matplotlib
import numpy as np

def test_setRcParams():
    hp.setRcParams()
    expected_paras = {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": ["sans-serif"],
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
    checks = [matplotlib.rcParams[k] == v for k, v in expected_paras.items()]
    print([(matplotlib.rcParams[k], v) for k, v in expected_paras.items()])
    assert np.all(checks)

