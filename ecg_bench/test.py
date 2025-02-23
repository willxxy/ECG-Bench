from ecg_bench.utils.viz_utils import VizUtil
import numpy as np

viz = VizUtil()
signal = np.random.randn(12, 500)

viz.plot_2d_ecg(signal, 'test', './pngs/', 100)