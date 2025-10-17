import argparse
import torch

from ecg_bench.elms.build_llm import BuildLLM
from ecg_bench.elms.build_encoder import BuildEncoder
from ecg_bench.elms.build_encoder_llm import BuildEncoderLLM
from ecg_bench.utils.gpu_setup import is_main


class BuildELM:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def build_elm(self, llm_tokenizer):
        elm_components = None

        if self.args.llm:
            llm_components = BuildLLM(self.args, llm_tokenizer, llm_tokenizer.pad_token_id, llm_tokenizer.eos_token_id).build_llm()
            elm_components = llm_components
            if self.args.encoder:
                encoder_components = BuildEncoder(self.args).build_encoder()
                elm_components = BuildEncoderLLM(llm_components, encoder_components, self.args).build_encoder_llm()
        assert elm_components is not None, print("ELM Components is None")
        if self.args.elm_ckpt:
            self.load_elm_checkpoint(elm_components)
        return elm_components

    def load_elm_checkpoint(self, elm_components):
        elm_checkpoint = torch.load(self.args.elm_ckpt, map_location="cpu", weights_only=False)
        elm_components["elm"].load_state_dict(elm_checkpoint["model_state_dict"], strict=True)
        if is_main():
            print(f"Loaded ELM checkpoint from {self.args.elm_ckpt}")
