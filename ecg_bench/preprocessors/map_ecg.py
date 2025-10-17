from tqdm import tqdm
import os
import glob
from pathlib import Path
from datasets import load_dataset
import argparse

from ecg_bench.configs.constants import MAPPED_DATASETS
from ecg_bench.utils.file_manager import FileManager


class MapECG:
    """Main class for mapping external datasets to base datas"""

    def __init__(self, args: argparse.Namespace, fm: FileManager):
        self.args = args
        self.fm = fm
        self.available_ecgs = set()
        if self.args.map_data not in MAPPED_DATASETS:
            raise ValueError(f"Unsupported dataset: {self.args.map_data}")

    def map_data(self):
        if self.args.map_data == "ecg_bench_pulse":
            data = self._prepare_ecg_bench_pulse()
        elif self.args.map_data == "ecg_instruct_pulse":
            data = self._prepare_ecg_instruct_pulse()
        elif self.args.map_data == "pretrain_mimic":
            data = self._prepare_pretrain_mimic()
        elif self.args.map_data == "ecg_instruct_45k":
            data = self._prepare_ecg_instruct_45k()
        elif self.args.map_data == "ecg-qa_ptbxl":
            data = self._prepare_ecg_qa_ptb()
        elif self.args.map_data == "ecg-qa_mimic-iv-ecg":
            data = self._prepare_ecg_qa_mimic()
        elif self.args.map_data in ["ecg_grounding_pulse", "ecg_grounding", "ecg_grounding_test"]:
            data = self._prepare_ecg_grounding()

        if self.args.dev:
            data = data[:100]

        valid_instances = []
        for instance in tqdm(data, desc="Mapping external dataset"):
            ecg_path, text, name, preprocessed_dir = self._process_mapping_instance(instance)
            for i in range(100):
                if f"{ecg_path}_{i}" in self.available_ecgs:
                    valid_instances.append({
                        "ecg_path": f"{preprocessed_dir}/{ecg_path}_{i}.npy",
                        "text": text,
                        "name": name,
                    })

        print(f"Total instances for {self.args.map_data}: {len(data)}")
        print(f"Length of available ecgs: {len(self.available_ecgs)}")
        print(f"Valid instances: {len(valid_instances)}")
        self.fm.save_json(valid_instances, f"./data/{self.args.map_data}_mapped_{self.args.segment_len}.json")

    def _process_mapping_instance(self, instance):
        name = instance.get("name", "")

        if self.args.map_data in ["ecg_instruct_45k", "pretrain_mimic"]:
            text = instance["conversations"]
            ecg_path = "_".join(instance["ecg"].split("/"))
            preprocessed_dir = f"./data/mimic/preprocessed_{self.args.segment_len}_{self.args.target_sf}"

        elif self.args.map_data == "ecg_instruct_pulse":
            text = instance["conversations"]
            ecg_path, preprocessed_dir = self._get_ecg_instruct_pulse_path(instance)

        elif self.args.map_data in ["ecg-qa_mimic-iv-ecg", "ecg-qa_ptbxl"]:
            text = [instance["question_type"], instance["question"], instance["answer"]]
            ecg_path = "_".join(instance["ecg_path"][0].split("/")[2:])
            if self.args.map_data == "ecg-qa_ptbxl":
                preprocessed_dir = f"./data/ptb/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
            else:
                preprocessed_dir = f"./data/mimic/preprocessed_{self.args.segment_len}_{self.args.target_sf}"

        elif self.args.map_data == "ecg_bench_pulse":
            text = instance["conversations"]
            file_name = instance["file_name"]
            name = instance["name"]
            ecg_path, preprocessed_dir = self._get_ecg_bench_pulse_path(name, file_name)

        elif self.args.map_data in ["ecg_grounding_pulse", "ecg_grounding", "ecg_grounding_test"]:
            text = instance["conversations"]
            file_name = instance["ecg"]
            ecg_path, preprocessed_dir = self._get_ecg_grounding_path(file_name)

        return ecg_path, text, name, preprocessed_dir

    def _prepare_ecg_grounding(self):
        base_datasets = ["mimic"]
        if self.args.map_data == "ecg_grounding_pulse":
            base_datasets.append("ptb")
            base_datasets.append("code15")
            data = self.fm.open_json("./data/ecg_grounding/ECG_Grounding_30k.json")
        elif self.args.map_data == "ecg_grounding":
            data = self.fm.open_json("./data/ecg_grounding/grounding_train_30k.json")
        elif self.args.map_data == "ecg_grounding_test":
            data = self.fm.open_json("./data/ecg_grounding/ecg-grounding-test.json")
        for dataset in base_datasets:
            preprocessed_dir = f"./data/{dataset}/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
            self.available_ecgs.update(f.stem for f in Path(preprocessed_dir).glob("*"))
        return data

    def _prepare_ecg_bench_pulse(self):
        json_path = f"./data/{self.args.map_data}/ecg_bench_pulse_datasets.json"
        if self.fm.ensure_directory_exists(file=json_path):
            data = self.fm.open_json(json_path)
        else:
            data = self._setup_ecg_bench_pulse(json_path)

        for dataset in ["ptb", "code15", "csn", "cpsc"]:
            preprocessed_dir = f"./data/{dataset}/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
            self.available_ecgs.update(f.stem for f in Path(preprocessed_dir).glob("*"))
        return data

    def _prepare_ecg_instruct_pulse(self):
        for dataset in ["ptb", "mimic", "code15"]:
            preprocessed_dir = f"./data/{dataset}/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
            self.available_ecgs.update(f.stem for f in Path(preprocessed_dir).glob("*"))

        data = self.fm.open_json(f"./data/{self.args.map_data}/{self.args.map_data}.json")
        return data

    def _prepare_ecg_qa_ptb(self):
        preprocessed_dir = f"./data/ptb/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
        self.available_ecgs.update(f.stem for f in Path(preprocessed_dir).glob("*"))
        dataset_name = self.args.map_data.split("_")[1]
        paraphrased_jsons = glob.glob(f"./data/ecg-qa/output/{dataset_name}/paraphrased/*/*.json")
        template_jsons = glob.glob(f"./data/ecg-qa/output/{dataset_name}/template/*/*.json")
        path_to_all_jsons = paraphrased_jsons + template_jsons
        data = self.setup_ecg_qa(path_to_all_jsons)
        return data

    def _prepare_ecg_qa_mimic(self):
        preprocessed_dir = f"./data/mimic/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
        self.available_ecgs.update(f.stem for f in Path(preprocessed_dir).glob("*"))
        dataset_name = self.args.map_data.split("_")[1]
        paraphrased_jsons = glob.glob(f"./data/ecg-qa/output/{dataset_name}/paraphrased/*/*.json")
        template_jsons = glob.glob(f"./data/ecg-qa/output/{dataset_name}/template/*/*.json")
        path_to_all_jsons = paraphrased_jsons + template_jsons
        data = self.setup_ecg_qa(path_to_all_jsons)
        return data

    def _prepare_pretrain_mimic(self):
        preprocessed_dir = f"./data/mimic/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
        self.available_ecgs.update(f.stem for f in Path(preprocessed_dir).glob("*"))
        data = self.fm.open_json(f"./data/{self.args.map_data}/{self.args.map_data}.json")
        return data

    def _prepare_ecg_instruct_45k(self):
        preprocessed_dir = f"./data/mimic/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
        self.available_ecgs.update(f.stem for f in Path(preprocessed_dir).glob("*"))
        data = self.fm.open_json(f"./data/{self.args.map_data}/{self.args.map_data}.json")
        return data

    def _setup_ecg_bench_pulse(self, json_path):
        self.list_of_hf_datasets = ["cpsc-test", "csn-test-no-cot", "code15-test", "ptb-test", "ptb-test-report", "ecgqa-test"]
        data = []

        for name in tqdm(self.list_of_hf_datasets, desc="Loading ECGBench datasets"):
            dataset = load_dataset("PULSE-ECG/ECGBench", name=name, streaming=False, cache_dir="./../.huggingface")

            for item in dataset["test"]:
                conversations = item["conversations"]
                file_path = item["image_path"]
                file_name = file_path.split("/")[-1].split("-")[0]

                # Handle ecgqa-test special case
                if name == "ecgqa-test":
                    for conv in conversations:
                        if isinstance(conv.get("value"), list):
                            conv["value"] = "".join(conv["value"])

                data.append({
                    "file_path": file_path,
                    "file_name": file_name,
                    "conversations": conversations,
                    "name": name,
                })

        self.fm.save_json(data, json_path)
        return data

    def _get_ecg_grounding_path(self, file_name):
        base_dataset_name = file_name.split("/")[0]
        if base_dataset_name == "mimic-iv":
            preprocessed_dir = f"./data/mimic/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
            file_name = "_".join(file_name.split("/")[1:])
        elif base_dataset_name == "ecg_ptbxl_benchmarking":
            preprocessed_dir = f"./data/ptb/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
            file_name = "_".join(file_name.split("/")[3:])
        elif base_dataset_name == "code15":
            preprocessed_dir = f"./data/code15/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
            file_name = file_name.split("/")[-1]
        return file_name, preprocessed_dir

    def _get_ecg_bench_pulse_path(self, name, file_name):
        if name in ["ecgqa-test", "ptb-test-report", "ptb-test"]:
            preprocessed_dir = f"./data/ptb/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
            subfolder = file_name[:2] + "000"
            return f"records500_{subfolder}_{file_name}", preprocessed_dir
        if name == "cpsc-test":
            cpsc_paths = glob.glob("./data/cpsc/*/*/*.hea")
            cpsc_filename_to_path = {os.path.basename(path).split(".")[0]: path.replace(".hea", "") for path in cpsc_paths}
            preprocessed_dir = f"./data/cpsc/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
            ecg_path = cpsc_filename_to_path[file_name]
            return "_".join(ecg_path.split("/")), preprocessed_dir
        if name == "csn-test-no-cot":
            csn_paths = glob.glob("./data/csn/WFDBRecords/*/*/*.hea")
            csn_filename_to_path = {os.path.basename(path).split(".")[0]: path.replace(".hea", "") for path in csn_paths}
            preprocessed_dir = f"./data/csn/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
            ecg_path = csn_filename_to_path[file_name]
            return "_".join(ecg_path.split("/")), preprocessed_dir
        if name == "code15-test":
            preprocessed_dir = f"./data/code15/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
            return file_name.split("-")[0], preprocessed_dir

        return ""

    def _get_ecg_instruct_pulse_path(self, instance):
        parts = instance["image"].split("/")
        dataset_image_type = parts[0]
        filename = parts[-1]

        if dataset_image_type in ["mimic_v4", "mimic"]:
            dataset_image_type = "mimic"
            base_filename = filename.split("-")[0]
            path_to_file = "_".join(parts[1:-1] + [base_filename])
            ecg_path = f"files_{path_to_file}"
        elif dataset_image_type in ["ptb-xl"]:
            dataset_image_type = "ptb"
            record_number = filename.split("_")[0]
            record_number = f"{record_number}_hr"
            subfolder = record_number[:2] + "000"
            ecg_path = f"records500_{subfolder}_{record_number}"
        elif dataset_image_type in ["code15_v4"]:
            dataset_image_type = "code15"
            ecg_path = filename.split("-")[0]

        preprocessed_dir = f"./data/{dataset_image_type}/preprocessed_{self.args.segment_len}_{self.args.target_sf}"
        return ecg_path, preprocessed_dir

    def setup_ecg_qa(self, glob_paths, question_types=["single-verify", "single-choose", "single-query"]):
        data = []
        for fname in sorted(glob_paths):
            loaded_file = self.fm.open_json(fname)
            filtered_list = [item for item in loaded_file if item["question_type"] in question_types]
            data.extend(filtered_list)
        return data
