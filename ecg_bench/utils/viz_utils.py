import ecg_plot

class VizUtil:
    @staticmethod
    def plot_1d_ecg(ecg, title, save_path, sample_rate):
        ecg_plot.plot_1(ecg[0], title=title, sample_rate=sample_rate)
        ecg_plot.save_as_png(file_name=title, path=save_path)
    
    @staticmethod
    def plot_2d_ecg(ecg, title, save_path, sample_rate):
        ecg_plot.plot(ecg, title=title, sample_rate=sample_rate)
        ecg_plot.save_as_png(file_name=title, path=save_path)