import gradio as gr
import numpy as np
import argparse
import gc
import torch
import random
import numpy as np
import os
from huggingface_hub import login

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


def end2end_chat(user_message, ecg_file, chat_history):
    """
    Simulate an end2end chat inference turn.
    
    Parameters:
      - user_message (str): The latest message from the user.
      - ecg_file (gr.File): An optional file upload containing ECG data.
      - chat_history (list): A list of previous conversation turns (as (user, response) tuples).
    
    Returns:
      - (str, list): An empty string to clear the text input and the updated chat history.
    
    In a production setting, here you would:
        1. Process the ECG file into your (12, N) array.
        2. Prepend or incorporate diagnostic information (via your conversation template)
           and prepare the prompt using your End2EndECGChatDataset logic.
        3. Run your model's generate_chat (or similar) method.
        4. Append the model's reply to the history.
    """
    ecg_info = ""
    # Process the ECG file if provided
    if ecg_file is not None:
        try:
            # Try loading as numpy array (if a .npy file etc.)
            data = np.load(ecg_file.name)
            ecg_info = f" [ECG shape: {data.shape}]"
        except Exception as e:
            ecg_info = f" [ECG file error: {str(e)}]"
    
    # Append the (user) turn to the history
    user_input = user_message + ecg_info
    chat_history.append((user_input, None))
    
    # In a real system, use your model inference here.
    # For demonstration, we simulate a response that echoes the user input.
    response = f"Simulated Response: You said '{user_message}'" + ecg_info
    chat_history[-1] = (chat_history[-1][0], response)
    
    # Return an empty string to clear the textbox and update the chat history.
    return "", chat_history

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
    
    # Define custom CSS to enforce equal height for both input boxes.
    with gr.Blocks(css="""
    .big_box {
    height: 200px !important;
    }
    """) as demo:
        gr.Markdown("# End2End ECG Chat Demo")
        
        # State to hold the conversation history
        state = gr.State([])

        # Chatbot component to display the conversation history
        chatbot = gr.Chatbot()

        # Create a row with columns to hold the file upload, text input and send button.
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
            with gr.Column(scale=0.3):
                send_btn = gr.Button("Send")
        
        # Wire up submission via both hitting Enter in the textbox and clicking the send button.
        text_input.submit(
            fn=end2end_chat,
            inputs=[text_input, ecg_input, state],
            outputs=[text_input, chatbot]
        )
        send_btn.click(
            fn=end2end_chat,
            inputs=[text_input, ecg_input, state],
            outputs=[text_input, chatbot]
        )
        
    demo.launch(share=True)
    
if __name__ == "__main__":
    main(get_args())