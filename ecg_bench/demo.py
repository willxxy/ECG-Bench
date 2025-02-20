import gradio as gr
import numpy as np

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