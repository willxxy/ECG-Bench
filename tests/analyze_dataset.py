import pandas as pd


def analyze_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    print(df.head())
    print(df.columns)


if __name__ == "__main__":
    analyze_dataset("ecg_bench/data/mimic/machine_measurements.csv")
