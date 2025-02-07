import numpy as np


from ecg_bench.utils.conversation_utils import get_conv_template
from transformers import AutoTokenizer, AutoModelForCausalLM
from ecg_bench.utils.ecg_tokenizer_utils import ECGByteTokenizer
from ecg_bench.utils.dir_file_utils import FileManager
import bpe

llama_model = "meta-llama/llama-3.2-1b"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, cache_dir='../.huggingface')
llama_model = AutoModelForCausalLM.from_pretrained(llama_model, cache_dir='../.huggingface')

vocab, merges = FileManager.open_tokenizer('./data/tokenizer_3500_300000.pkl')
vocab_keys = list(vocab.keys())
# print(vocab_keys)
new_ids = [f'signal_{str(ids)}' for ids in vocab_keys]
llama_tokenizer.add_tokens(new_ids)
llama_tokenizer.add_special_tokens({"pad_token":"<pad>"})
llama_model.config.pad_token_id = llama_tokenizer.pad_token_id
# llama_model.resize_token_embeddings(len(llama_tokenizer)) # skip for now

# Add all the special tokens you need
new_special_tokens = {
    'additional_special_tokens': [
        '<|start_header_id|>',
        '<|end_header_id|>',
        '<|eot_id|>'
    ]
}

# Add these special tokens to the tokenizer
num_added_tokens = llama_tokenizer.add_special_tokens(new_special_tokens)

print("Special tokens mapping:", llama_tokenizer.special_tokens_map)
print("\nAll special tokens:", llama_tokenizer.all_special_tokens)
print("\nAll special token IDs:", llama_tokenizer.all_special_ids)

# If you're using a model, don't forget to resize its token embeddings
# llama_model.resize_token_embeddings(len(llama_tokenizer))

print(f"\nNumber of tokens added: {num_added_tokens}")

altered_text_multiturn = [
        {
            "from": "human",
            "value": "<ecg>\nWhat is the rhythm shown in the ECG?"
        },
        {"from": "gpt", "value": "Sinus rhythm."},
        {
            "from": "human",
            "value": "Are there any signs of right bundle branch block?"
        },
        {"from": "gpt", "value": "No, there are no signs of right bundle branch block."},
        {
            "from": "human",
            "value": "What is the peak amplitude of the T wave?"
        },
        {
            "from": "gpt",
            "value": "The T wave peak is 2617 mV."
        },
        {"from": "human", "value": "Is the ECG normal?"},
        {"from": "gpt", "value": "Yes, the ECG is normal."},
    ]

original_text_multiturn = altered_text_multiturn.copy()
altered_text_multiturn = []

for message in original_text_multiturn:
    if message["from"] == "human" and "<ecg>" in message["value"]:
        # For messages containing <ecg>, keep the tag for later replacement
        altered_text_multiturn.append(message)
    else:
        # For other messages, just add them as is
        altered_text_multiturn.append(message)

conv = get_conv_template("llama-3")
conv.set_system_message("You are a helpful assistant that can see and understand images. Please provide detailed and accurate responses.")

# Build the conversation and get the prompt
for message in altered_text_multiturn:
    role = conv.roles[0] if message['from'] == 'human' else conv.roles[1]
    conv.append_message(role, message['value'])

prompt = conv.get_prompt()
print(prompt)

# Find the position of <ecg> in the prompt
ecg_position = prompt.find("<ecg>")

# Split the prompt into before and after <ecg>
prompt_before_ecg = prompt[:ecg_position]
prompt_after_ecg = prompt[ecg_position + len("<ecg>"):]

# Tokenize the parts separately
tokens_before = llama_tokenizer.encode(prompt_before_ecg, add_special_tokens=False)
tokens_after = llama_tokenizer.encode(prompt_after_ecg, add_special_tokens=False)

# Calculate available space for ECG signal
pad_to_max = 2048
conversation_len = len(tokens_before) + len(tokens_after)
available_space = pad_to_max - conversation_len

symbols = list('abcdefghijklmnopqrstuvwxyz')

ecg_signal = np.random.rand(12, 500)
clipped_normalized = np.clip(ecg_signal, 0, 1)
scaled_signal = np.minimum(
    np.floor(clipped_normalized * len(symbols)),
    len(symbols) - 1
).astype(np.uint8)
symbol_signal = np.vectorize(lambda x: symbols[x])(scaled_signal)
symbol_signal = ''.join(symbol_signal.flatten())
encoded_signal = bpe.encode_symbol(symbol_signal, FileManager.open_tokenizer('./data/tokenizer_3500_300000.pkl')[1])
tokenized_signal = llama_tokenizer.convert_tokens_to_ids([f'signal_{ids}' for ids in encoded_signal])
print(len(tokenized_signal))

# Truncate or pad the ECG signal as needed
print('tokenized signal', len(tokenized_signal))
if len(tokenized_signal) > available_space:
    print(f"Warning: Signal truncated from {len(tokenized_signal)} to {available_space} tokens")
    ecg_tokens = tokenized_signal[:available_space]
else:
    ecg_tokens = tokenized_signal

print('ecg tokens', len(ecg_tokens))

# Combine all parts
final_tokens = tokens_before + ecg_tokens + tokens_after

labels = [-100] * len(final_tokens)

for i, message in enumerate(altered_text_multiturn):
    if message['from'] == 'gpt':
        # Get the assistant's message
        response = message['value']
        # Find the response in the decoded text
        response_tokens = llama_tokenizer.encode(response, add_special_tokens=False)
        # Find these tokens in final_tokens and set their labels
        for j in range(len(final_tokens) - len(response_tokens) + 1):
            if final_tokens[j:j+len(response_tokens)] == response_tokens:
                labels[j:j+len(response_tokens)] = response_tokens
                
eot_id = llama_tokenizer.convert_tokens_to_ids('<|eot_id|>')
for i, token_id in enumerate(final_tokens):
    if token_id == eot_id:
        labels[i] = eot_id
        
print('final tokens', len(final_tokens))
if len(final_tokens) < pad_to_max:
    padding_length = pad_to_max - len(final_tokens)
    final_tokens = [llama_tokenizer.pad_token_id] * padding_length + final_tokens
    labels = [-100] * padding_length + labels

print('final tokens', len(final_tokens))
print(len(final_tokens), len(labels))

assert len(final_tokens) == pad_to_max, f"Expected length {pad_to_max}, got {len(final_tokens)}"
assert len(final_tokens) == len(labels), "Tokens and labels length mismatch"

print(final_tokens)

tokens = llama_tokenizer.convert_ids_to_tokens(final_tokens)

decoded_text = llama_tokenizer.decode(final_tokens, skip_special_tokens=False)
print("\nDecoded text:")
print(decoded_text)

# print("\nTokenized prompt with corresponding token IDs:")
# for idx, (token, token_id) in enumerate(zip(tokens, final_tokens)):
#         print(f"{idx}: {token} -> {token_id}")