import ecg_plot
import matplotlib.pyplot as plt

class VizUtil:
    @staticmethod
    def plot_1d_ecg(ecg, title, save_path, sample_rate):
        ecg_plot.plot_1(ecg[0], title=title, sample_rate=sample_rate)
        ecg_plot.save_as_png(file_name=title, path=save_path)
    
    @staticmethod
    def plot_2d_ecg(ecg, title, save_path, sample_rate):
        ecg_plot.plot(ecg, title=title, sample_rate=sample_rate)
        ecg_plot.save_as_png(file_name=title, path=save_path)
        
    @staticmethod
    def plot_train_val_loss(train_loss, val_loss = None, dir_path = None):
        epochs = range(1, len(train_loss) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, 'b', label='Training loss')
        if val_loss is not None:
            plt.plot(epochs, val_loss, 'r', label='Validation loss')
            plt.title('Training and Validation Loss')
        else:
            plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{dir_path}/train_val_loss.png')
        plt.close()