import ecg_plot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import numpy as np
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
    def get_plot_as_image(ecg, sample_rate, title = ''):
        return ecg_plot.get_plot_as_image(ecg, title=title, sample_rate=sample_rate)
        
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
    
    @staticmethod        
    def plot_attention_on_signal(signal, attention_weights, lead, count):
        signal = signal[lead]
        attention_weights = attention_weights[lead]
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.plot(signal, label='Signal', linewidth=2)
        ax2 = ax.twinx()
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.fill_between(range(len(attention_weights)), 0, attention_weights, 
                        color='orange', alpha=0.7, label='Attention')
        ax2.set_ylim(0, 0.03)
        ax.set_xlabel('Sequence Length', fontsize=18)
        ax.set_ylabel('Signal Amplitude', fontsize=18)
        ax.set_title('Distribution of Attention Weights Across ECG')
        ax2.set_ylabel('Attention Weight', fontsize=18)
        plt.tight_layout()
        plt.savefig(f'./pngs/sig_att_{lead}_{count}.png')
        plt.close()
        
    @staticmethod
    def plot_text_attention_weights(tokens, attention_weights, count, figsize=(12, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams.update({'font.size': 18})
        x = np.arange(len(attention_weights))
        ax.bar(x, attention_weights, color='orange', alpha=0.7, label='Attention')
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_xlabel('Text Tokens')
        ax.set_ylabel('Attention Weight')
        ax.set_title('Distribution of Attention Weights Across Text')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(0, 0.03)
        plt.tight_layout()
        plt.savefig(f'./pngs/question_and_answer_{count}.png')
        plt.close()
        
    @staticmethod
    def plot_distributions(token_counts, token_lengths, vocab_size):
        plt.figure(figsize=(12, 5))
        plt.rcParams.update({'font.size': 18})
        plt.subplot(1, 2, 1)
        token_freq = sorted(token_counts.values(), reverse=True)
        plt.plot(range(1, len(token_freq) + 1), token_freq)
        plt.title('Token Usage Distribution', fontsize=16)
        plt.xlabel('Token Rank', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.yscale('log')
        plt.xscale('log')
        plt.subplot(1, 2, 2)
        plt.hist(token_lengths, bins=30, edgecolor='black')
        plt.title('Token Length Distribution', fontsize=16)
        plt.xlabel('Number of Tokens', fontsize=14)
        plt.xlim(0, max(token_lengths))
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'./pngs/token_dist_ind_{vocab_size}.png')
        plt.close()
        
    @staticmethod
    def generate_distinct_colors(n):
        palette = sns.color_palette("husl", n)
        return [tuple(rgb) for rgb in palette]

    @staticmethod
    def visualize_bpe_encoding(signal, encoded_ids, segment_map,
                               lead_name=None, id_to_color=None, count = None):
        plt.figure(figsize=(20, 8))
        plt.rcParams.update({
            'font.size': 25,
            'axes.titlesize': 25,
            'axes.labelsize': 25,
            'legend.fontsize': 25,
            'xtick.labelsize': 25,
            'ytick.labelsize': 25
        })    
        for id, (start, end) in zip(encoded_ids, segment_map):
            plt.axvspan(start, end, color=id_to_color[id], alpha=1)
        plt.plot(signal, color='black', linewidth=3.0, alpha=1.0)
        plt.title(f'Token IDs Overlayed on ECG Lead {lead_name}', pad=20)
        plt.xlabel('Time', labelpad=12)
        plt.ylabel('Amplitude', labelpad=12) # make the scales the same across all graphs
        plt.ylim(0, 1)
        
        present_ids = sorted(set(encoded_ids))
        legend_elements = [Patch(facecolor=id_to_color[id], edgecolor='black', label=f'ID: {id}')
                        for id in present_ids]
        
        plt.legend(handles=legend_elements,
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                title='Token IDs',
                title_fontsize=18,
                ncol=2)
        
        plt.tight_layout()
        plt.savefig(f'./pngs/ecg_encoding_visualization_{lead_name}_{count}.png',
                    bbox_inches='tight',
                    dpi=300)
        plt.close()