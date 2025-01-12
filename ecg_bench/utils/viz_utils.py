import ecg_plot

class VizUtil:
    @staticmethod
    def plot_1d_ecg(ecg, title, save_path, sample_rate):
        ecg = VizUtil._reorder_indices(ecg)
        ecg_plot.plot_1(ecg[0], title=title, sample_rate=sample_rate)
        ecg_plot.save_as_png(file_name=title, path=save_path)
    
    @staticmethod
    def plot_2d_ecg(ecg, title, save_path, sample_rate):
        ecg = VizUtil._reorder_indices(ecg)
        ecg_plot.plot(ecg, title=title, sample_rate=sample_rate)
        ecg_plot.save_as_png(file_name=title, path=save_path)

    @staticmethod
    def _reorder_indices(ecg):
        current_order = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        desired_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        order_mapping = {lead: index for index, lead in enumerate(current_order)}
        new_indices = [order_mapping[lead] for lead in desired_order]
        return ecg[new_indices, :]