import gradio as gr
import numpy as np
import argparse
import gc
import torch
torch.set_num_threads(6)
import random
import os
from huggingface_hub import login
from functools import partial
import PIL.Image
import io
import base64
import json  # for saving the structured log

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
    data_group.add_argument('--image', action = 'store_true', default=None, help='Turn Image Generation on')
    data_group.add_argument('--augment_image', action = 'store_true', default=None, help='Turn Image Augmentation on')
    data_group.add_argument('--instance_normalize', action = 'store_true', default=True, help='Turn Instance Normalization on')
    
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
    mode_group.add_argument('--post_train', action='store_true', default=None, help='Post-training mode')
    model_group.add_argument('--train_encoder', action='store_true', default=None, help='Train encoder too')
    mode_group.add_argument('--interpret', action='store_true', default=None, help='Interpret mode')
    mode_group.add_argument('--rag', action='store_true', default=None, help='RAG mode')
    
    ### Checkpoints and Paths
    ckpt_group = parser.add_argument_group('Checkpoints')
    ckpt_group.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    ckpt_group.add_argument('--encoder_checkpoint', type=str, default=None, help='Encoder checkpoint path')

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
            
            train_utils.viz.plot_2d_ecg(data, "test", "./pngs/", sample_rate=250)
            img = PIL.Image.open("./pngs/test.png")
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            img_markdown = f'<img src="data:image/png;base64,{img_str}" style="width:100%; max-width:100%;">'
            
            symbol_signal = train_utils.ecg_tokenizer_utils._to_symbol_string(data)
            encoded_signal = train_utils.ecg_tokenizer_utils.encode_symbol(
                symbol_signal, train_utils.ecg_tokenizer_utils.merges
            )
            list_of_signal_tokens = [f'signal_{ids}' for ids in encoded_signal]
            joined_signal_tokens = ''.join(list_of_signal_tokens)
            full_user_message = joined_signal_tokens + '\n' + user_message
            display_user_message = strip_hidden_tokens(full_user_message)
        except Exception as e:
            print(str(e))
            full_user_message = user_message
            display_user_message = user_message
    else:
        img_markdown = None
        full_user_message = user_message
        display_user_message = user_message

    conv.append_message(conv.roles[0], full_user_message)
    prompt = conv.get_prompt()
    tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(tokenized_prompt, dtype=torch.int64)
    attention_mask = create_attention_mask(tokenized_prompt, pad_id)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float32)
    assert input_ids.shape[0] == attention_mask.shape[0]

    response = model.generate_demo(
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer
    )
    conv.append_message(conv.roles[1], response)

    if img_markdown is not None:
        assistant_response = img_markdown + "\n\n" + response
    else:
        assistant_response = response

    # Update the chat display history.
    state["history"].append((display_user_message, assistant_response))
    
    # Also update the structured log.
    if "structured_history" not in state or state["structured_history"] is None:
        state["structured_history"] = {}
    turn_number = len(state["structured_history"]) + 1
    state["structured_history"][f"turn_{turn_number}"] = {
        "user_input": display_user_message,
        "assistant_response": response,
        "preference": None,  # No preference given yet.
        "ecg_input" : "Yes" if ecg_file is not None else "No"
    }

    print('Full prompt:\n', conv.get_prompt())
    print('--------------------------------')
    return "", state["history"], None

# Callback functions for preference buttons.
def record_feedback(feedback, state):
    if "structured_history" in state and state["structured_history"]:
        last_turn_key = f"turn_{len(state['structured_history'])}"
        # Only record feedback if none exists for this turn
        if state["structured_history"][last_turn_key]["preference"] is None:
            state["structured_history"][last_turn_key]["preference"] = feedback
            # Save the structured log to a file
            with open("./data/conversation_history.json", "w") as f:
                json.dump(state["structured_history"], f, indent=2)
            return state, f"✓ Feedback recorded: {feedback}"
        return state, "Feedback already recorded for this response"
    return state, ""

def record_good(state):
    return record_feedback("good", state)

def record_bad(state):
    return record_feedback("bad", state)

def main(args):
    device = setup_environment(args)
    fm, viz = initialize_system(args)
    
    ecg_tokenizer_utils = ECGByteTokenizer(args, fm)
    train_utils = TrainingUtils(args, fm, viz, device, ecg_tokenizer_utils)
    
    print(f'Creating Model: {args.model}')
    model_object = train_utils.create_model()
    
    model = model_object['llm']
    if args.checkpoint != None:
        checkpoint = torch.load(f"{args.checkpoint}/best_model.pth", map_location=device)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')
    else:
        print('No checkpoint provided')
    tokenizer = model_object['llm_tokenizer']
    
    chat_fn = partial(end2end_chat, model=model, tokenizer=tokenizer, train_utils=train_utils)
    
    with gr.Blocks(css="""
    .big_box {
        height: 200px !important;
    }
    .chatbox {
        height: 600px !important;
    }
    .feedback_message {
        text-align: center;
        color: #4CAF50;
        margin-top: 5px;
    }
    """) as demo:
        gr.Markdown("# End2End ECG Chat Demo")
        # Initialize state with conversation objects
        state = gr.State({"conv": None, "history": [], "structured_history": {}})
        
        chatbot = gr.Chatbot(elem_classes="chatbox")
        
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
        
        # Feedback row right below chatbox
        with gr.Row():
            with gr.Column(scale=1):
                good_btn = gr.Button("👍 Good")
                bad_btn = gr.Button("👎 Bad")
            feedback_message = gr.Markdown(elem_classes="feedback_message")
        
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
        
        good_btn.click(record_good, inputs=[state], outputs=[state, feedback_message])
        bad_btn.click(record_bad, inputs=[state], outputs=[state, feedback_message])
        
    demo.launch(share=True)
    
if __name__ == "__main__":
    main(get_args())
