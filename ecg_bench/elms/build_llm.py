import argparse
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

from ecg_bench.configs.constants import HF_LLMS, HF_CACHE_DIR
from ecg_bench.elms.llm.hf_llm import HuggingFaceLLM
from ecg_bench.utils.gpu_setup import is_main


class BuildLLM:
    def __init__(self, args: argparse.Namespace, llm_tokenizer, pad_token_id, eos_token_id):
        self.args = args
        self.llm_tokenizer = llm_tokenizer
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def build_llm(
        self,
    ):
        if self.args.llm in HF_LLMS:
            llm = self.build_hf()
        else:
            raise ValueError(f"{self.args.llm} not supported.")
        llm = HuggingFaceLLM(llm, self.pad_token_id, self.eos_token_id)
        if self.args.dev and is_main():
            self.print_llm_dtype(llm)
        key_name = "llm" if self.args.encoder is not None else "elm"
        components = {
            key_name: llm,
            "find_unused_parameters": HF_LLMS[self.args.llm]["find_unused_parameters"],
        }
        return components

    ### HF FUNCTIONS ###
    def build_hf(
        self,
    ):
        hf_llm = self.build_hf_llm()
        HF_LLMS[self.args.llm]["model_hidden_size"] = hf_llm.config.hidden_size
        assert HF_LLMS[self.args.llm]["model_hidden_size"] is not None, print("model_hidden_size")
        hf_llm = self.resize_and_report_embeddings(hf_llm)
        if self.args.peft:
            hf_llm = self.build_peft(
                hf_llm,
            )
        return hf_llm

    ### HF LLM FUNCTIONS ###
    def build_hf_llm(
        self,
    ):
        return AutoModelForCausalLM.from_pretrained(
            HF_LLMS[self.args.llm]["model"],
            cache_dir=HF_CACHE_DIR,
            dtype=HF_LLMS[self.args.llm]["native_dtype"],
            attn_implementation=self.args.attention_type,
        )

    def print_llm_dtype(self, llm):
        print(
            f"{self.args.llm} native dtype:", HF_LLMS[self.args.llm]["native_dtype"], f"\n{self.args.llm} actual dtype:", next(llm.parameters()).dtype
        )
        assert HF_LLMS[self.args.llm]["native_dtype"] == next(llm.parameters()).dtype, print(f"{self.args.llm} native and actual dtype do not match")

    def resize_and_report_embeddings(self, hf_llm):
        old_size = hf_llm.get_input_embeddings().weight.shape[0]
        if is_main():
            print(f"[{self.args.llm}] embedding size before: {old_size}")
        hf_llm.resize_token_embeddings(len(self.llm_tokenizer))
        new_size = hf_llm.get_input_embeddings().weight.shape[0]
        if is_main():
            print(f"[{self.args.llm}] embedding size after: {new_size}")
        assert new_size == len(self.llm_tokenizer), f"Embedding size {new_size} does not match tokenizer vocab size {len(self.llm_tokenizer)}"
        return hf_llm

    ### PEFT FUNCTIONS ###
    def build_peft(
        self,
        llm,
    ):
        lora_config = self.get_lora_configs()
        llm = get_peft_model(llm, lora_config)
        if is_main():
            llm.print_trainable_parameters()
        return llm

    def get_lora_configs(
        self,
    ):
        lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            task_type=TaskType.CAUSAL_LM,
        )
        return lora_config
