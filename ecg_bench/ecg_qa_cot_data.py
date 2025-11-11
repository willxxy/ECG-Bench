#!/usr/bin/env python3
"""
Script to display the first data item from ECG QA CoT dataset as JSON
"""

import csv
import json
import sys


def show_first_data(csv_file):
    """
    Read the first data item from CSV and display as JSON

    Args:
        csv_file: Path to the CSV file
    """
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            first_row = next(reader)

            # Pretty print as JSON
            print(json.dumps(first_row, indent=2, ensure_ascii=False))

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found", file=sys.stderr)
        sys.exit(1)
    except StopIteration:
        print("Error: CSV file is empty", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Default path to the ECG QA CoT training data
    csv_file = "data/ecg-qa-cot/ecg_qa_cot/ecg_qa_cot_train.csv"

    # Allow custom file path as command line argument
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]

    show_first_data(csv_file)