import pandas as pd
import numpy as np
import os
import glob
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse

from ecg_bench.configs.constants import BASE_DATASETS
from ecg_bench.utils.file_manager import FileManager


class PrepareDF:
    """Main class for preparing all base datas in a easy to understand dataframe format"""

    def __init__(self, args: argparse.Namespace, fm: FileManager):
        self.args = args
        self.fm = fm
        if self.args.base_data not in BASE_DATASETS:
            raise ValueError(f"Unsupported dataset: {self.args.base_data}")

    def prepare_df(self):
        print("Preparing dataframe...")
        if self.args.base_data == "ptb":
            df = self.prepare_ptb_df()
        elif self.args.base_data == "mimic":
            df = self.prepare_mimic_df()
        elif self.args.base_data == "code15":
            df = self.prepare_code15_df()
        elif self.args.base_data == "cpsc":
            df = self.prepare_cpsc_df()
        elif self.args.base_data == "csn":
            df = self.prepare_csn_df()
        print("Dataframe prepared.")
        print("Saving dataframe...")
        df.to_csv(f"./data/{self.args.base_data}/{self.args.base_data}.csv", index=False)

    def get_df(self):
        print("Getting dataframe...")
        df = pd.read_csv(f"./data/{self.args.base_data}/{self.args.base_data}.csv")
        print("Dataframe retrieved.")
        print("Cleaning dataframe...")
        df = self.fm.clean_dataframe(df)
        print("Dataframe cleaned.")
        if self.args.dev:
            print("Dev mode is on. Reducing dataframe size to 1000 instances...")
            df = df.iloc[:1000]
        if self.args.toy:
            print("Toy mode is on. Reducing dataframe size to 60% of original size...")
            df = df.sample(frac=0.60, random_state=42).reset_index(drop=True)
        print("Dataframe retrieved and cleaned.")
        print(df.head())
        print("Number of instances in dataframe:", len(df))
        print("Dataframe prepared.")
        return df

    def prepare_ptb_df(self):
        ptbxl_database = pd.read_csv("./data/ptb/ptbxl_database.csv", index_col="ecg_id")
        ptbxl_database = ptbxl_database.rename(columns={"filename_hr": "path"})
        df = ptbxl_database[["path", "report"]]
        df = self.translate_german_to_english(df)
        return df

    def prepare_mimic_df(self):
        record_list = pd.read_csv("./data/mimic/record_list.csv")
        machine_measurements = pd.read_csv("./data/mimic/machine_measurements.csv")
        report_columns = [f"report_{i}" for i in range(18)]
        machine_measurements["report"] = machine_measurements[report_columns].apply(
            lambda x: " ".join([str(val) for val in x if pd.notna(val)]), axis=1
        )
        mm_columns = ["subject_id", "study_id"] + report_columns + ["report"]

        merged_df = pd.merge(
            record_list[["subject_id", "study_id", "file_name", "path"]], machine_measurements[mm_columns], on=["subject_id", "study_id"], how="inner"
        )

        merged_df = merged_df.dropna(subset=report_columns, how="all")
        df = merged_df[["path", "report"]]
        return df

    def prepare_code15_df(self):
        exam_mapping = self._build_code15_exam_mapping()
        df = pd.DataFrame([
            {
                "exam_id": exam_id,
                "path": file_path,
                "idx": idx,
                "report": "placeholder report",  # Empty report column to match other datasets
            }
            for exam_id, (file_path, idx) in exam_mapping.items()
        ])
        return df

    def prepare_cpsc_df(self):
        hf_dataset = load_dataset("PULSE-ECG/ECGBench", name="cpsc-test", streaming=False, cache_dir="./../.huggingface")
        cpsc_paths = glob.glob("./data/cpsc/*/*/*.hea")
        cpsc_filename_to_path = {os.path.basename(path).split(".")[0]: path.replace(".hea", "") for path in cpsc_paths}
        df = pd.DataFrame([])
        for item in hf_dataset["test"]:
            file_path = item["image_path"]
            file_name = file_path.split("/")[-1].split("-")[0]
            conversations = item["conversations"]
            if file_name in cpsc_filename_to_path:
                new_row = pd.DataFrame({
                    "path": [cpsc_filename_to_path[file_name]],
                    "report": [conversations],
                    "orig_file_name": [file_name],
                })
                df = pd.concat([df, new_row], ignore_index=True)
        return df

    def prepare_csn_df(self):
        hf_dataset = load_dataset("PULSE-ECG/ECGBench", name="csn-test-no-cot", streaming=False, cache_dir="./../.huggingface")
        csn_paths = glob.glob("./data/csn/WFDBRecords/*/*/*.hea")
        csn_filename_to_path = {os.path.basename(path).split(".")[0]: path.replace(".hea", "") for path in csn_paths}
        df = pd.DataFrame([])
        for item in hf_dataset["test"]:
            file_path = item["image_path"]
            file_name = file_path.split("/")[-1].split("-")[0]
            conversations = item["conversations"]
            if file_name in csn_filename_to_path:
                new_row = pd.DataFrame({
                    "path": [csn_filename_to_path[file_name]],
                    "report": [conversations],
                    "orig_file_name": [file_name],
                })
                df = pd.concat([df, new_row], ignore_index=True)
        return df

    def _build_code15_exam_mapping(self):
        import h5py

        mapping = {}
        for part in range(18):
            file_path = f"./data/code15/exams_part{part}.hdf5"
            with h5py.File(file_path, "r") as f:
                # Need to add patient id for stratification.
                exam_ids = f["exam_id"][:]
                for idx, eid in enumerate(exam_ids):
                    if isinstance(eid, bytes):
                        eid = eid.decode("utf-8")
                    eid = str(int(eid))
                    mapping[eid] = (file_path, idx)
        return mapping

    def translate_german_to_english(self, df):
        texts = df["report"].values
        try:
            if isinstance(texts, list):
                texts = np.array(texts)

            if not isinstance(texts, np.ndarray):
                raise ValueError("Input must be a numpy array or list")
            if texts.ndim != 1:
                raise ValueError(f"Expected 1D array, got shape {texts.shape}")
            if len(texts) == 0:
                raise ValueError("Input array cannot be empty")

            valid_mask = np.array([bool(text and str(text).strip()) for text in texts])
            valid_texts = texts[valid_mask]

            if len(valid_texts) == 0:
                raise ValueError("All input texts are empty")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en", cache_dir="./../.huggingface")
            model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en", cache_dir="./../.huggingface").to(device)

            batch_size = 64
            translations = []

            for i in tqdm(range(0, len(valid_texts), batch_size), desc="Translating files"):
                batch_texts = valid_texts[i : i + batch_size]

                encoded = tokenizer(list(batch_texts), return_tensors="pt", padding=True, truncation=True)
                encoded = {key: tensor.to(device) for key, tensor in encoded.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **encoded,
                        max_length=128,
                    )

                batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                translations.extend(batch_translations)

            result = np.empty_like(texts, dtype=object)
            result[valid_mask] = translations
            result[~valid_mask] = ""

            translated_df = df.copy()
            translated_df["report"] = result

            return translated_df

        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Translation error: {e!s}")
