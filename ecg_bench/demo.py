import gradio as gr
import numpy as np
import argparse
import gc
import torch
torch.set_num_threads(6)
import random
import numpy as np
import os
from huggingface_hub import login
from functools import partial

from ecg_bench.utils.conversation_utils import get_conv_template
from ecg_bench.utils.dir_file_utils import FileManager
from ecg_bench.utils.viz_utils import VizUtil
from ecg_bench.utils.training_utils import TrainingUtils
from ecg_bench.utils.ecg_tokenizer_utils import ECGByteTokenizer

def get_args():
    parser = argparse.ArgumentParser(description = None)
    
    ### Data
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data', type=str, default=None, help='Dataset name')
    data_group.add_argument('--seg_len', type=int, default=500, help='Segment length')
    data_group.add_argument('--num_merges', type=int, default=3500, help='Vocab size')
    data_group.add_argument('--target_sf', type=int, default=250, help='Target sampling frequency')
    data_group.add_argument('--pad_to_max', type=int, default=1020, help='Pad to max size')
    data_group.add_argument('--ecg_tokenizer', type=str, help='Tokenizer specification')
    data_group.add_argument('--percentiles', type=str, default=None, help='Percentiles computed during preprocessing')
    data_group.add_argument('--system_prompt', type=str, default=None, help='System prompt')
    
    ### Model
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--model', type=str, default=None, help='Model name')
    model_group.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    model_group.add_argument('--seed', type=int, default=0, help='Random seed')
    
    ### Optimizer
    optim_group = parser.add_argument_group('Optimizer')
    optim_group.add_argument('--attn_implementation', type=str, default=None, help='Attention implementation')
    
    ### PEFT
    peft_group = parser.add_argument_group('PEFT')
    peft_group.add_argument('--peft', action='store_true', default=None, help='Use PEFT')
    peft_group.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
    peft_group.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    peft_group.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    
    ### Mode and Environment
    mode_group = parser.add_argument_group('Mode and Environment')
    mode_group.add_argument('--dev', action='store_true', default=None, help='Development mode')
    mode_group.add_argument('--inference', type=str, default = 'end2end', choices=['second', 'end2end'], help='Inference mode')
    mode_group.add_argument('--train', type=str, default = None, choices=['first', 'second', 'end2end'], help='Training mode')
    mode_group.add_argument('--interpret', action='store_true', default=None, help='Interpret mode')
    
    ### Checkpoints and Paths
    ckpt_group = parser.add_argument_group('Checkpoints')
    ckpt_group.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    ckpt_group.add_argument('--encoder_checkpoint', type=str, default=None, help='Encoder checkpoint path')
    ckpt_group.add_argument('--encoder_data', type=str, default=None, help='Encoder data path')

    return parser.parse_args()

def setup_environment(args):
    print('Setting up Single Device')
    device = torch.device(args.device)
    args.device = device
    return device

def initialize_system(args):
    print('Loading API key')
    with open('./../.huggingface/api_keys.txt', 'r') as file:
        api_key = file.readlines()[0].strip()
    login(token=api_key)
    
    print('System Cleanup')
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f'Setting Seed to {args.seed}')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.dev:
        print('Running in Development Mode')
        args.epochs = 2
        args.log = False
        # args.batch_size = 1  # Changed from 2 to 1
    
    return FileManager(), VizUtil()

def create_attention_mask(input_ids, pad_id):
    return [0 if num == pad_id else 1 for num in input_ids]


@torch.no_grad()
def end2end_chat(user_message, ecg_file, state, model, tokenizer, train_utils):
    # Helper function to remove tokens with the specified prefix.
    def strip_hidden_tokens(message, token_prefix="signal_"):
        lines = message.split('\n')
        filtered_lines = [line for line in lines if not line.startswith(token_prefix)]
        return "\n".join(filtered_lines)

    model.eval()
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    if state["conv"] is None:
        conv = get_conv_template('llama-3')
        system_prompt = (
            "You are an expert multimodal assistant capable of processing both natural language text and ECG signals. "
            "When you receive input, first determine if it is text, ECG data, or both. For ECG signals, interpret them as "
            "time-series data representing cardiac activity—analyzing features such as heart rate, rhythm, and potential abnormalities. "
            "When both modalities are present, synthesize the information to provide integrated, expert cardiac electrophysiologist-level responses. "
            "Your answers should be precise, concise, and informed by clinical signal analysis and natural language understanding. "
            "Additionally, if the user asks a general question, you should answer it as a general assistant."
        )
        conv.set_system_message(system_prompt)
        state["conv"] = conv
    else:
        conv = state["conv"]

    # Process ECG file if provided.
    if ecg_file is not None:
        try:
            data = np.load(ecg_file.name)
            symbol_signal = train_utils.ecg_tokenizer_utils._to_symbol_string(data)
            encoded_signal = train_utils.ecg_tokenizer_utils.encode_symbol(
                symbol_signal, train_utils.ecg_tokenizer_utils.merges
            )
            list_of_signal_tokens = [f'signal_{ids}' for ids in encoded_signal]
            joined_signal_tokens = ''.join(list_of_signal_tokens)
            # Create the full user message with signal tokens for model input.
            full_user_message = joined_signal_tokens + '\n' + user_message
            # Create a display message that removes the signal tokens.
            display_user_message = strip_hidden_tokens(full_user_message)
        except Exception as e:
            print(str(e))
            full_user_message = user_message
            display_user_message = user_message
    else:
        full_user_message = user_message
        display_user_message = user_message

    # Append full user message to the conversation for model inference.
    conv.append_message(conv.roles[0], full_user_message)

    # Get the full prompt for inference.
    prompt = conv.get_prompt()
    tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(tokenized_prompt, dtype=torch.int64)
    attention_mask = create_attention_mask(tokenized_prompt, pad_id)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float32)
    assert input_ids.shape[0] == attention_mask.shape[0]

    # Simulate a response.
    response = model.generate_demo(
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer
    )

    # Append the assistant's response.
    conv.append_message(conv.roles[1], response)

    # Record the conversation history using the display (UI-friendly) version.
    state["history"].append((display_user_message, response))

    print('Full prompt:\n', conv.get_prompt())
    print('--------------------------------')
    return "", state["history"], None

def main(args):
    device = setup_environment(args)
    fm, viz = initialize_system(args)
    
    ecg_tokenizer_utils = ECGByteTokenizer(args, fm)
    train_utils = TrainingUtils(args, fm, viz, device, ecg_tokenizer_utils)
    
    print(f'Creating Model: {args.model}')
    model_object = train_utils.create_model()
    
    model = model_object['llm']
    tokenizer = model_object['llm_tokenizer']
    
    # checkpoint = torch.load(f"{args.checkpoint}/best_model.pth", map_location=args.device)
    # model.load_state_dict(checkpoint['model'])
    # print('Model loaded')
    
    chat_fn = partial(end2end_chat, model=model, tokenizer=tokenizer, train_utils=train_utils)
    
    # Define custom CSS to enforce equal height for both input boxes.
    with gr.Blocks(css="""
    .big_box {
        height: 200px !important;
    }
    """) as demo:
        gr.Markdown("# End2End ECG Chat Demo")
        
        # Initialize state as a dictionary to persist conversation.
        state = gr.State({"conv": None, "history": []})
        
        chatbot = gr.Chatbot()
        
        with gr.Row():
            with gr.Column(scale=1):
                ecg_input = gr.File(
                    label="Optional ECG file (.npy, .dat, .hea)", 
                    file_types=[".npy", ".dat", ".hea"],
                    elem_classes="big_box"
                )
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    elem_classes="big_box"
                )
            with gr.Column(scale=1):
                send_btn = gr.Button("Send")
        

        text_input.submit(
            fn=chat_fn,
            inputs=[text_input, ecg_input, state],
            outputs=[text_input, chatbot, ecg_input]
        )
        send_btn.click(
            fn=chat_fn,
            inputs=[text_input, ecg_input, state],
            outputs=[text_input, chatbot, ecg_input]
        )

        
    demo.launch(share=True)
    
if __name__ == "__main__":
    main(get_args())