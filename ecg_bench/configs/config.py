import argparse
from ecg_bench.configs.constants import Mode


def get_args(mode: Mode) -> argparse.Namespace:
    if mode not in {"train", "eval", "inference", "post_train", "ecg_tokenizer", "preprocess", "rag"}:
        raise ValueError(f"invalid mode: {mode}")

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    parser.add_argument("--dev", action="store_true", default=None, help="Development mode")
    if mode in {"train", "eval", "inference", "post_train", "ecg_tokenizer"}:
        parser.add_argument("--ecg_tokenizer", type=str, default=None, help="Path to ECG Tokenizer")
    if mode in {"train", "eval", "inference", "post_train", "preprocess"}:
        parser.add_argument("--segment_len", type=int, default=1250, help="ECG Segment Length")
    if mode in {"train", "eval", "inference", "post_train"}:
        for flag, help_text in [
            ("--ecg_image", "Plot ECG Signal as Image"),
            ("--ecg_signal", "Raw ECG Signal"),
            ("--ecg_stacked_signal", "Stacked ECG Signal"),
            ("--ecg_token", "ECG Tokens"),
            ("--augment_ecg_image", "Augment ECG Image"),
            ("--noise_ecg", "Apply ECG Perturbation"),
            ("--blackout_ecg", "Apply ECG Blackout"),
            ("--no_signal", "No signal, text only"),
        ]:
            parser.add_argument(flag, action="store_true", default=None, help=help_text)

        parser.add_argument("--data", type=str, default=None, help="ID of the training/eval/inference/post-train data from huggingface datasets")
        parser.add_argument("--encoder", type=str, default=None, help="Neural Network Encoder Model")
        parser.add_argument("--llm", type=str, default=None, help="Large Language Model")
        parser.add_argument("--peft", action="store_true", default=None, help="Use PEFT")
        parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
        parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
        parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
        parser.add_argument("--encoder_ckpt", type=str, default=None, help="Path to the encoder checkpoint")
        parser.add_argument("--elm_ckpt", type=str, default=None, help="Path to the LLM checkpoint")
        parser.add_argument("--attention_type", type=str, default="sdpa", help="Attention Type")
        parser.add_argument("--num_encoder_tokens", type=int, default=1, help="Number of encoder tokens")
        parser.add_argument("--update_encoder", action="store_true", default=False, help="Update encoder")

        parser.add_argument("--system_prompt", type=str, default=None, help="Path to System Prompt")
        parser.add_argument("--fold", type=str, default="1", help="Data Fold Number")

        parser.add_argument("--rag", action="store_true", default=None, help="Use RAG")
        parser.add_argument("--rag_k", type=int, default=1, help="RAG k")
        parser.add_argument("--rag_database", type=str, default=None, help="Path to RAG Database Containing Metadata and Indexes")
        parser.add_argument("--rag_query", type=str, default=None, choices=["ecg_signal", "ecg_feature"], help="Rag Query Type")
        parser.add_argument("--rag_location", type=str, default="system_prompt", choices=["system_prompt", "user_query"], help="RAG Location")
        parser.add_argument("--rag_content", type=str, default=None, choices=["ecg_feature", "diagnostic_report"], help="RAG Content")

        parser.add_argument("--wandb", action="store_true", default=None, help="Enable logging")

        parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
        parser.add_argument("--distributed", action="store_true", default=None, help="Enable distributed training")

    if mode == "train":
        parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"], help="Optimizer type")
        parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
        parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
        parser.add_argument("--patience_delta", type=float, default=0.1, help="Delta for early stopping")
        parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for optimizer")
        parser.add_argument("--beta2", type=float, default=0.99, help="Beta2 for optimizer")
        parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for optimizer")
        parser.add_argument("--warmup", type=int, default=500, help="Warmup steps")
        parser.add_argument("--ref_global_bs", type=int, default=None)
        parser.add_argument("--grad_accum_steps", type=int, default=1)
        parser.add_argument("--scale_wd", type=str, default="none", choices=["none", "inv_sqrt", "inv_linear"])
        parser.add_argument("--llm_input_len", type=int, default=1024, help="LLM Input Sequence Length")
        parser.add_argument("--min_ecg_tokens_len", type=int, default=250, help="Minimum ECG token length to consider")

    if mode == "post_train":
        parser.add_argument("--dpo_beta", type=float, default=0.5, help="DPO beta")

    if mode in {"ecg_tokenizer", "preprocess"}:
        parser.add_argument("--num_cores", type=int, default=12, help="Number of cores for parallel processing")

    if mode == "ecg_tokenizer":
        parser.add_argument("--num_merges", type=int, default=3500, help="Number of merges for BPE")
        parser.add_argument("--num_samples", type=int, default=300000, help="Number of samples for training the tokenizer")
        parser.add_argument("--sampled_file", type=str, default=None, help="Path to the sampled ECG files for tokenizer training")
        parser.add_argument("--path_to_ecg_npy", type=str, default=None, help="Path to the ECG npy files")

    if mode == "preprocess":
        parser.add_argument("--preprocess", action="store_true", default=False, help="Preprocess files")
        parser.add_argument("--base_data", type=str, default=None, help="Base dataset to preprocess")
        parser.add_argument("--map_data", type=str, default=None, help="External dataset to map to base dataset")
        parser.add_argument("--toy", action="store_true", default=None, help="Create a toy dataset")
        parser.add_argument("--mix_data", type=str, default=None, help="Mix data: comma-separated list of JSON filenames")
        parser.add_argument("--target_sf", type=int, default=250, help="Target sampling frequency")

    if mode == "rag":
        parser.add_argument("--rag_data", type=str, default=None, help="Path to the data for RAG database creation")

    return parser.parse_args()
