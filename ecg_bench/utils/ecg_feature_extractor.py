import numpy as np
import argparse
from scipy import signal
import pywt


class ECGFeatureExtractor:
    def __init__(
        self,
        args: argparse.Namespace,
    ):
        self.args = args

    def extract_features(self, ecg):
        features = []

        for lead in range(ecg.shape[0]):
            lead_signal = ecg[lead, :]

            features.extend([
                np.mean(lead_signal),
                np.std(lead_signal),
                np.max(lead_signal),
                np.min(lead_signal),
                np.median(lead_signal),
                np.percentile(lead_signal, 25),
                np.percentile(lead_signal, 75),
            ])

            freqs, psd = signal.welch(lead_signal, fs=self.args.target_sf, nperseg=min(1024, len(lead_signal)))
            total_power = np.sum(psd)
            features.extend([
                total_power,
                np.max(psd),
                freqs[np.argmax(psd)],
            ])

            if total_power > 0:
                spectral_centroid = np.sum(freqs * psd) / total_power
            else:
                spectral_centroid = 0
            features.append(spectral_centroid)

            if np.max(lead_signal) != np.min(lead_signal):
                peak_height = 0.3 * (np.max(lead_signal) - np.min(lead_signal)) + np.min(lead_signal)
                min_distance = max(int(0.2 * self.args.target_sf), 1)
                peaks, _ = signal.find_peaks(lead_signal, height=peak_height, distance=min_distance)
            else:
                peaks = []

            heart_rate_features = self._calculate_heart_rate_features(lead_signal, peaks)
            features.extend(heart_rate_features)

            wavelet_features = self._calculate_wavelet_features(lead_signal)
            features.extend(wavelet_features)

            features.append(np.mean(np.abs(np.diff(lead_signal))))
            features.append(np.sqrt(np.mean(np.square(np.diff(lead_signal)))))

        return np.array(features)

    def _calculate_heart_rate_features(self, ecg, peaks):
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / self.args.target_sf
            heart_rate = 60 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0

            hrv = np.std(rr_intervals) if len(rr_intervals) > 1 else 0

            qrs_duration = np.mean([self.find_qrs_duration(ecg, peak) for peak in peaks])

            t_wave_amp = self.find_t_wave_amplitude(ecg, peaks)

            st_deviation = self.find_st_deviation(ecg, peaks)

            return [heart_rate, hrv, qrs_duration, t_wave_amp, st_deviation]
        return [0, 0, 0, 0, 0]

    def _calculate_wavelet_features(self, signal):
        try:
            max_level = min(5, pywt.dwt_max_level(len(signal), "db4"))
            coeffs = pywt.wavedec(signal, "db4", level=max_level)
            return [np.mean(np.abs(c)) for c in coeffs]
        except Exception:
            return [0] * 6

    def find_qrs_duration(self, ecg, peak):
        window = int(0.1 * self.args.target_sf)
        start = max(0, peak - window)
        end = min(len(ecg), peak + window)
        qrs_segment = ecg[start:end]
        if len(qrs_segment) == 0 or np.max(qrs_segment) == np.min(qrs_segment):
            return 0
        threshold = 0.1 * (np.max(qrs_segment) - np.min(qrs_segment)) + np.min(qrs_segment)
        return np.sum(np.abs(qrs_segment - np.mean(qrs_segment)) > threshold) / self.args.target_sf

    def find_t_wave_amplitude(self, ecg, peaks):
        if len(peaks) < 2:
            return 0
        start_idx = peaks[-2]
        end_idx = min(peaks[-1], len(ecg) - 1)
        if start_idx >= end_idx or start_idx < 0:
            return 0
        t_wave_region = ecg[start_idx:end_idx]
        return np.max(t_wave_region) - np.min(t_wave_region) if len(t_wave_region) > 0 else 0

    def find_st_deviation(self, ecg, peaks):
        if len(peaks) < 1 or peaks[-1] >= len(ecg):
            return 0
        st_offset = int(0.08 * self.args.target_sf)
        st_point = min(peaks[-1] + st_offset, len(ecg) - 1)
        if st_point < len(ecg):
            return ecg[st_point] - ecg[peaks[-1]]
        return 0
