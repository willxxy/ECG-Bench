import glob

from ecg_bench.utils.viz_utils import VizUtil
from ecg_bench.utils.dir_file_utils import FileManager


fm = FileManager()
viz = VizUtil()

mimic_pre_500_250 = glob.glob('data/mimic/preprocessed_500_250/*.npy')
mimic_pre_2500_250 = glob.glob('data/mimic/preprocessed_2500_250/*.npy')

ptb_pre_500_250 = glob.glob('data/ptb/preprocessed_500_250/*.npy')

print(len(mimic_pre_500_250), len(mimic_pre_2500_250), len(ptb_pre_500_250))
ecg_array_500_250 = fm.open_npy(mimic_pre_500_250[0])
ecg_array_2500_250 = fm.open_npy(mimic_pre_2500_250[0])
ecg_array_500_250_ptb = fm.open_npy(ptb_pre_500_250[0])

print(ecg_array_500_250['ecg'].shape, ecg_array_2500_250['ecg'].shape, ecg_array_500_250_ptb['ecg'].shape)
viz.plot_2d_ecg(ecg_array_500_250['ecg'], 'mimic_500_250_2d', './pngs/', 250)
viz.plot_1d_ecg(ecg_array_500_250['ecg'], 'mimic_500_250_1d', './pngs/', 250)

viz.plot_2d_ecg(ecg_array_2500_250['ecg'], 'mimic_2500_250_2d', './pngs/', 250)
viz.plot_1d_ecg(ecg_array_2500_250['ecg'], 'mimic_2500_250_1d', './pngs/', 250)

viz.plot_2d_ecg(ecg_array_500_250_ptb['ecg'], 'ptb_500_250_2d', './pngs/', 250)
viz.plot_1d_ecg(ecg_array_500_250_ptb['ecg'], 'ptb_500_250_1d', './pngs/', 250)