from openai import OpenAI
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from ecg_bench.utils.file_manager import FileManager

QRS_GRAPH = {
    "QRS": {"question": "Is a QRS complex present?", "choices": ["Yes", "No (Asystole)"]},
    "Pacing": {"question": "Is pacing present?", "choices": ["Yes", "No"]},
    "Axis": {
        "question": "What is the axis of the QRS complex?",
        "choices": [
            "normal",
            "Right/LPFB",
            "Left/LAFB",
            "NW",
        ],
    },
    "Lead reversal": {
        "question": "Is there a lead reversal present?",
        "choices": [
            "LA/RA",
            "LA/LL",
            "RA/RL",
        ],
    },
    "Rate": {
        "question": "What is the rate of the QRS complex?",
        "choices": [
            "Bradycardia",
            "Tachycardia",
            "Normal",
        ],
    },
    "Amplitude": {
        "question": "What is the amplitude of the QRS complex?",
        "choices": [
            "Normal",
            "LVH/RVH",
            "Low",
        ],
    },
    "Preexcitation": {"question": "Is there a preexcitation present?", "choices": ["Yes", "No"]},
    "AP": {
        "question": "What is the AP?",
        "choices": [
            "Normal",
            "Right/LPFB",
            "Left/LAFG",
            "NW",
        ],
    },
    "Duration": {
        "question": "What is the duration of the QRS complex?",
        "choices": [
            "<110",
            ">120",
            "110-120",
        ],
    },
    ">120": {
        "question": "If the duration is >120, please specify:",
        "choices": [
            "IVCD",
            "RBB",
            "LBBB",
        ],
    },
    "110-120": {
        "question": "If the duration is 110-120, please specify:",
        "choices": [
            "iLBBB",
            "iRBB",
            "Other",
        ],
    },
    "<110": {
        "question": "If the duration is <110, please specify:",
        "choices": [
            "rSR'",
            "None",
        ],
    },
}

T_GRAPH = {
    "T": {
        "question": "What is the morphology of the T wave?",
        "choices": [
            "Normal",
            "Peaked",
            "Inverted",
            "Nonspecific",
        ],
    },
}

NOISE_ARTIFACTS_GRAPH = {
    "Noise artifacts": {
        "question": "What kind of noise artifacts are present?",
        "choices": [
            "Missing",
            "LVAD",
            "Noise",
        ],
    },
}

QRS_QUESTION_ORDER = ["QRS", "Pacing", "Axis", "Lead reversal", "Rate", "Amplitude", "Preexcitation", "AP", "Duration"]
NOISE_ARTIFACTS_QUESTION_ORDER = ["Noise artifacts"]
T_QUESTION_ORDER = ["T"]
ALL_QUESTION_ORDER = QRS_QUESTION_ORDER + NOISE_ARTIFACTS_QUESTION_ORDER + T_QUESTION_ORDER

SECTIONS = [
    ("QRS", QRS_GRAPH),
    ("Noise Artifacts", NOISE_ARTIFACTS_GRAPH),
    ("T", T_GRAPH),
]

SYSTEM_PROMPT = """You are an expert ECG interpreter.
Given an ECG diagnosis, you will answer a series of structured questions about the ECG characteristics.

You must respond with a valid JSON object containing answers to all questions. Follow these rules:
1. Answer each question with ONE choice from the provided options
2. For conditional questions (like ">120", "110-120", "<110"), only answer if the parent condition is met
3. If a conditional question doesn't apply, use null as the value
4. All answers must be from the provided choices only
5. Return ONLY valid JSON with no additional text

The JSON structure must follow this format:
{
    "QRS": "choice",
    "Pacing": "choice",
    "Axis": "choice",
    "Lead reversal": "choice or null",
    "Rate": "choice",
    "Amplitude": "choice",
    "Preexcitation": "choice",
    "AP": "choice",
    "Duration": "choice",
    ">120": "choice or null",
    "110-120": "choice or null",
    "<110": "choice or null",
    "Noise artifacts": "choice",
    "T": "choice"
}

If the diagnosis does NOT clearly imply one of these mappings, leave the corresponding field as null.
"""


def create_user_prompt(diagnosis: str) -> str:
    # Build the questions structure
    questions_text = []

    for section_name, section_graph in SECTIONS:
        questions_text.append(f"\n## {section_name} Section")
        for key in ALL_QUESTION_ORDER:
            if key in section_graph:
                q_data = section_graph[key]
                choices_str = ", ".join([f'"{c}"' for c in q_data["choices"]])
                questions_text.append(f"\n{key}: {q_data['question']}")
                questions_text.append(f"  Choices: [{choices_str}]")

    questions_section = "\n".join(questions_text)

    user_prompt = f"""Given the ECG diagnosis: "{diagnosis}"

Answer the following questions about this ECG:

{questions_section}

Respond with a JSON object containing your answers."""

    return user_prompt


DATASETS = [
    # "ecg-qa-ptbxl-250-2500",
    "ecg-qa-mimic-iv-ecg-250-2500",
]
FILE_MANAGER = FileManager()

print("=" * 60)
print("TESTING SIMPLE GENERATION")
print("=" * 60)

client = OpenAI()
model = "gpt-5-mini"

output_img = False
lead_names = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

prompt_format = create_user_prompt("EXAMPLE_DIAGNOSIS")

output_data = {
    "prompts": {
        "system_prompt": SYSTEM_PROMPT,
        "prompt_format": prompt_format,
    },
    "results": [],
}

dataset = load_dataset(f"willxxy/{DATASETS[0]}", split="fold1_train").with_transform(FILE_MANAGER.decode_batch)
max_instances = 100

for step, data in enumerate(tqdm(dataset)):
    if step >= max_instances:
        break

    ecg_path = data["ecg_path"]
    ecg_path = ecg_path.replace("./data", "./ecg_bench/data")
    ecg_np_file = FILE_MANAGER.open_npy(ecg_path)
    diagnostic = ecg_np_file["report"]
    user_prompt = create_user_prompt(diagnostic)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        output_data["results"].append({
            "diagnostic": diagnostic,
            "response": result,
        })
    except json.JSONDecodeError as e:
        print(f"JSON decode error at step {step}: {e}. Skipping this instance.")
        continue
    except Exception as e:
        print(f"Error at step {step}: {e}. Skipping this instance.")
        continue

output_path = "cot_results.json"
FILE_MANAGER.save_json(output_data, output_path)
print(f"Saved results to {output_path}")
print(f"Total instances processed: {len(output_data['results'])}")
