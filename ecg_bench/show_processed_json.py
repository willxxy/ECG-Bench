import json

# Read the first entry from the JSON file
with open('./data/ecg_qa_cot_mapped_1250.json', 'r') as f:
    data = json.load(f)

    # Display the first entry
    if isinstance(data, list) and len(data) > 0:
        print("Processed ECG-QA-COT:")
        print(json.dumps(data[0], indent=2))
    else:
        print(f"Unexpected data type: {type(data)}")

with open('./data/ecg-qa_ptbxl_mapped_1250.json', 'r') as f:
    data = json.load(f)

    # Display the first entry
    if isinstance(data, list) and len(data) > 0:
        print("Processed ECG-QA-PTBXL:")
        print(json.dumps(data[0], indent=2))
    else:
        print(f"Unexpected data type: {type(data)}")
