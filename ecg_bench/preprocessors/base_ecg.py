import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import numpy as np
from scipy import interpolate
from tqdm import tqdm


class BaseECG:
    """Main class for preprocessing all base datas"""

    def __init__(self, args, fm, df):
        self.args = args
        self.fm = fm
        self.df = df
        self.preprocessed_dir = f"./data/{self.args.base_data}/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
        self.fm.ensure_directory_exists(folder=self.preprocessed_dir)

    def preprocess_batch(self):
        skipped_count = 0
        try:
            with ProcessPoolExecutor(max_workers=self.args.num_cores) as executor:
                futures = [executor.submit(self._process_single_instance, idx) for idx in range(len(self.df))]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing ECGs..."):
                    try:
                        result = future.result()
                        if result is None:
                            skipped_count += 1
                    except Exception as e:
                        print(f"Error processing instance: {e!s}")
                        skipped_count += 1
        except Exception as e:
            print(f"Error in preprocess_instance: {e!s}")
        finally:
            print(f"Total instances skipped: {skipped_count}")

    def _process_single_instance(self, idx):
        save_dic = {}
        try:
            if self.args.base_data == "mimic":
                file_path = f"./data/mimic/{self.df.iloc[idx]['path']}"
                report = self.df.iloc[idx]["report"]
                ecg, sf = self.fm.open_ecg(file_path)
                assert sf == 500 and ecg.shape == (5000, 12)
            elif self.args.base_data == "ptb":
                file_path = f"./data/ptb/{self.df.iloc[idx]['path']}"
                report = self.df.iloc[idx]["report"]
                ecg, sf = self.fm.open_ecg(file_path)
                assert sf == 500 and ecg.shape == (5000, 12)
            elif self.args.base_data == "code15":
                file_path = f"{self.df.iloc[idx]['path']}"
                report = self.df.iloc[idx]["report"]
                tracing_idx = self.df.iloc[idx]["idx"]
                exam_id = self.df.iloc[idx]["exam_id"]
                sf = 400  # code15 has a sampling frequency of 400 Hz
                with h5py.File(file_path, "r") as f:
                    ecg = f["tracings"][tracing_idx]
                assert ecg.shape == (4096, 12) and sf == 400
            elif self.args.base_data == "cpsc":
                file_path = f"{self.df.iloc[idx]['path']}"
                report = self.df.iloc[idx]["report"]
                ecg, sf = self.fm.open_ecg(file_path)
                assert sf == 500  # cant assert shape because the shape is variable
            elif self.args.base_data == "csn":
                file_path = f"{self.df.iloc[idx]['path']}"
                report = self.df.iloc[idx]["report"]
                ecg, sf = self.fm.open_ecg(file_path)
                assert sf == 500 and ecg.shape == (5000, 12)

            if self.args.base_data == "mimic" or self.args.base_data == "code15":
                ecg = self._reorder_indices(ecg)

            if sf != self.args.target_sf:
                downsampled_ecg = self.nsample_ecg(ecg, orig_fs=sf, target_fs=self.args.target_sf)
            else:
                downsampled_ecg = ecg

            downsampled_ecg = downsampled_ecg.astype(np.float32)

            orig_dur = downsampled_ecg.shape[0] / self.args.target_sf
            segmented_ecg, segmented_text = self.segment_ecg(downsampled_ecg, report, segment_len=self.args.segment_len)
            seg_dur = self.args.segment_len / self.args.target_sf

            assert len(segmented_text) == segmented_ecg.shape[0]
            segmented_ecg = self._check_nan_inf(segmented_ecg, "preprocessing")

            if np.any(np.isnan(segmented_ecg)) or np.any(np.isinf(segmented_ecg)):
                print(f"Warning: NaN values detected in {file_path}. Skipping this instance.")
                return None

            if orig_dur != seg_dur:
                for j in range(len(segmented_text)):
                    save_dic = {
                        "ecg": np.transpose(segmented_ecg[j], (1, 0)),
                        "report": segmented_text[j],
                        "path": self.df.iloc[idx]["path"],
                        "orig_sf": sf,
                        "target_sf": self.args.target_sf,
                        "segment_len": self.args.segment_len,
                    }
                    if self.args.base_data == "code15":
                        save_dic["exam_id"] = exam_id
                        save_dic["tracing_idx"] = tracing_idx
                        save_path = f"{self.preprocessed_dir}/{exam_id}_{j}.npy"
                    else:
                        save_path = f"{self.preprocessed_dir}/{'_'.join(self.df.iloc[idx]['path'].split('/'))}_{j}.npy"
                    if self._check_save_dictionary(save_dic):  # Check if dictionary is valid
                        np.save(save_path, save_dic)
                        if not self._verify_saved_file(save_path):  # Quick verify the save was successful
                            print(f"Failed to save file properly: {save_path}")
                            return None
            else:
                save_dic = {
                    "ecg": np.transpose(segmented_ecg[0], (1, 0)),
                    "report": segmented_text[0],
                    "path": self.df.iloc[idx]["path"],
                    "orig_sf": sf,
                    "target_sf": self.args.target_sf,
                    "segment_len": self.args.segment_len,
                }
                if self.args.base_data == "code15":
                    save_dic["exam_id"] = exam_id
                    save_dic["tracing_idx"] = tracing_idx
                    save_path = f"{self.preprocessed_dir}/{exam_id}_0.npy"
                else:
                    save_path = f"{self.preprocessed_dir}/{'_'.join(self.df.iloc[idx]['path'].split('/'))}_0.npy"
                if self._check_save_dictionary(save_dic):  # Check if dictionary is valid
                    np.save(save_path, save_dic)
                    if not self._verify_saved_file(save_path):  # Quick verify the save was successful
                        print(f"Failed to save file properly: {save_path}")
                        return None

            return True

        except Exception as e:
            print(f"Error processing {file_path}: {e!s}. Skipping this instance.")
            return None

    def _check_nan_inf(self, ecg, step_name):
        if np.any(np.isnan(ecg)) or np.any(np.isinf(ecg)):
            print(f"Warning: NaN or inf values detected after {step_name}")
            ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
        return ecg

    def _verify_saved_file(self, save_path):
        try:
            if not os.path.exists(save_path):
                return False
            if os.path.getsize(save_path) == 0:
                os.remove(save_path)  # Remove empty file
                return False
            return True
        except Exception as e:
            print(f"Error verifying saved file {save_path}: {e!s}")
            if os.path.exists(save_path):
                os.remove(save_path)  # Remove corrupted file
            return False

    def _check_save_dictionary(self, save_dic):
        if not save_dic or len(save_dic) == 0:
            return False
        if "ecg" not in save_dic or save_dic["ecg"].size == 0:
            return False
        if "report" not in save_dic or not save_dic["report"]:
            return False
        return True

    def _reorder_indices(self, ecg):
        if self.args.base_data == "mimic":
            current_order = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
        elif self.args.base_data == "code15":
            current_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        desired_order = ["I", "II", "III", "aVL", "aVR", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        order_mapping = {lead: index for index, lead in enumerate(current_order)}
        new_indices = [order_mapping[lead] for lead in desired_order]
        return ecg[:, new_indices]

    def nsample_ecg(self, ecg, orig_fs, target_fs):
        num_samples, num_leads = ecg.shape
        duration = num_samples / orig_fs
        t_original = np.linspace(0, duration, num_samples, endpoint=True)
        t_target = np.linspace(0, duration, int(num_samples * target_fs / orig_fs), endpoint=True)

        downsampled_data = np.zeros((len(t_target), num_leads))
        for lead in range(num_leads):
            f = interpolate.interp1d(t_original, ecg[:, lead], kind="cubic", bounds_error=False, fill_value="extrapolate")
            downsampled_data[:, lead] = f(t_target)
        return downsampled_data

    def segment_ecg(self, ecg, report, segment_len):
        time_length, _ = ecg.shape
        num_segments = time_length // segment_len

        ecg_data_segmented = []
        text_data_segmented = []

        for i in range(num_segments):
            start_idx = i * segment_len
            end_idx = (i + 1) * segment_len
            ecg_data_segmented.append(ecg[start_idx:end_idx, :])
            text_data_segmented.append(report)

        return np.array(ecg_data_segmented), text_data_segmented
