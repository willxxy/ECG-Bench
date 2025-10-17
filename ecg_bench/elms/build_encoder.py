import argparse
from transformers import AutoModel, ViTForMaskedImageModeling
import torch
import numpy as np

from ecg_bench.configs.constants import VISION_ENCODERS, ECG_ENCODERS, HF_CACHE_DIR, VISION_ENCODERS_INPUT_MAPPING
from ecg_bench.utils.gpu_setup import is_main, GPUSetup
from ecg_bench.elms.vision_encoder.hf_encoder import HuggingFaceEncoder


class BuildEncoder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.gpu_setup = GPUSetup(args)
        self.device = self.gpu_setup.get_device()

    def build_encoder(
        self,
    ):
        if self.args.encoder in VISION_ENCODERS:
            encoder_components = self.build_hf_encoder()
            encoder_components["find_unused_parameters"] = VISION_ENCODERS[self.args.encoder]["find_unused_parameters"]
            assert VISION_ENCODERS[self.args.encoder]["model_hidden_size"] is not None, print(f"{self.args.encoder} model_hidden_size is None")
            if self.args.dev and is_main():
                self.print_encoder_dtype(encoder_components["encoder"])
        elif self.args.encoder in ECG_ENCODERS:
            encoder_components = self.build_non_hf_encoder()
            encoder_components["find_unused_parameters"] = ECG_ENCODERS[self.args.encoder]["find_unused_parameters"]
            assert ECG_ENCODERS[self.args.encoder]["model_hidden_size"] is not None, print(f"{self.args.encoder} model_hidden_size is None")
        else:
            raise ValueError(f"{self.args.encoder} not supported.")
        if self.args.encoder_ckpt:
            self.load_encoder_checkpoint(encoder_components)
        return encoder_components

    ### NON HF ENCODER FUNCTIONS ###
    def build_non_hf_encoder(
        self,
    ):
        if self.args.encoder == "projection":
            return self.build_projection()
        elif self.args.encoder == "merl":
            return self.build_merl()
        elif self.args.encoder == "mtae":
            return self.build_mtae()
        elif self.args.encoder == "mlae":
            return self.build_mlae()
        elif self.args.encoder == "st_mem":
            return self.build_st_mem()
        else:
            raise ValueError(f"{self.args.encoder} not supported.")

    def build_merl(
        self,
    ):
        if self.args.encoder and self.args.llm:
            from ecg_bench.elms.ecg_encoder.merl import MERLFinetune

            ecg_encoder = MERLFinetune(ECG_ENCODERS[self.args.encoder]["model"], self.args.num_encoder_tokens)
        else:
            from ecg_bench.elms.ecg_encoder.merl import MERLPretrain

            if self.args.segment_len == 1250:
                ECG_ENCODERS[self.args.encoder]["spacial_dim"] = 79
            elif self.args.segment_len == 2500:
                ECG_ENCODERS[self.args.encoder]["spacial_dim"] = 157
            else:
                ECG_ENCODERS[self.args.encoder]["spacial_dim"] = 32
            assert ECG_ENCODERS[self.args.encoder]["spacial_dim"] is not None, print(f"{self.args.encoder} spacial_dim is None")

            lm = AutoModel.from_pretrained(ECG_ENCODERS[self.args.encoder]["tokenizer"], cache_dir=HF_CACHE_DIR)
            ecg_encoder = MERLPretrain(ECG_ENCODERS[self.args.encoder]["model"], lm, self.args.encoder, self.args.distributed)
        return {"encoder": ecg_encoder}

    def build_st_mem(
        self,
    ):
        from ecg_bench.elms.ecg_encoder.st_mem import ST_MEM_Ours, st_mem_vit_base_dec256d4b

        ecg_encoder = ST_MEM_Ours(
            st_mem_vit_base_dec256d4b(device=self.device, num_leads=12, seq_len=self.args.segment_len, patch_size=self.calculate_patch_size()),
            num_encoder_tokens=self.args.num_encoder_tokens,
        )
        return {"encoder": ecg_encoder}

    def build_mtae(
        self,
    ):
        from ecg_bench.elms.ecg_encoder.mtae import MTAE_Ours, mtae_vit_base_dec256d4b

        ecg_encoder = MTAE_Ours(
            mtae_vit_base_dec256d4b(device=self.device, num_leads=12, seq_len=self.args.segment_len, patch_size=self.calculate_patch_size()),
            num_encoder_tokens=self.args.num_encoder_tokens,
        )
        return {"encoder": ecg_encoder}

    def build_mlae(
        self,
    ):
        from ecg_bench.elms.ecg_encoder.mlae import MLAE_Ours, mlae_vit_base_dec256d4b

        ecg_encoder = MLAE_Ours(
            mlae_vit_base_dec256d4b(device=self.device, num_leads=12, seq_len=self.args.segment_len, patch_size=1),
            num_encoder_tokens=self.args.num_encoder_tokens,
        )
        return {"encoder": ecg_encoder}

    def build_projection(
        self,
    ):
        ECG_ENCODERS[self.args.encoder]["model_hidden_size"] = 12 * self.args.segment_len
        ECG_ENCODERS[self.args.encoder]["projection_dim"] = 12 * self.args.segment_len
        from ecg_bench.elms.ecg_encoder.projection import Projection

        if self.args.encoder in VISION_ENCODERS:
            projection_dim = VISION_ENCODERS[self.args.encoder]["projection_dim"]
        else:
            projection_dim = ECG_ENCODERS[self.args.encoder]["projection_dim"]
        ecg_encoder = Projection(projection_dim, self.args.llm)
        return {"encoder": ecg_encoder}

    def calculate_patch_size(self):
        min_patches = 16
        max_patches = 64
        factors = [i for i in range(1, self.args.segment_len + 1) if self.args.segment_len % i == 0]
        patch_candidates = []
        for patch_size in factors:
            num_patches = self.args.segment_len // patch_size
            if min_patches <= num_patches <= max_patches:
                patch_candidates.append(patch_size)
        if not patch_candidates:
            target = int(np.sqrt(self.args.segment_len / 32))
            patch_size = min(factors, key=lambda x: abs(x - target))
        else:
            patch_size = min(patch_candidates, key=lambda x: abs(self.args.segment_len // x - 32))
        return patch_size

    ### HF ENCODER FUNCTIONS ###
    def build_hf_encoder(
        self,
    ):
        if self.args.encoder == "vit-base-patch16-224-in21k":
            hf_encoder = ViTForMaskedImageModeling.from_pretrained(VISION_ENCODERS[self.args.encoder]["model"], cache_dir=HF_CACHE_DIR)
        else:
            hf_encoder = AutoModel.from_pretrained(VISION_ENCODERS[self.args.encoder]["model"], cache_dir=HF_CACHE_DIR)
        if self.args.encoder == "clip-vit-base-patch32":
            embed_out, extra_configs = self.build_hf_clip(hf_encoder)
        elif self.args.encoder == "siglip-base-patch16-224":
            embed_out, extra_configs = self.build_hf_siglip(hf_encoder)
        elif self.args.encoder == "vit-base-patch16-224-in21k":
            embed_out, extra_configs = self.build_hf_vit(hf_encoder)
        hf_encoder = HuggingFaceEncoder(
            hf_encoder,
            VISION_ENCODERS_INPUT_MAPPING[self.args.encoder],
            extra_configs,
            embed_out if bool(self.args.llm) and bool(self.args.encoder) else None,
        )
        return {"encoder": hf_encoder}

    def build_hf_clip(self, hf_encoder: AutoModel):
        VISION_ENCODERS[self.args.encoder]["model_hidden_size"] = hf_encoder.config.projection_dim
        VISION_ENCODERS[self.args.encoder]["projection_dim"] = hf_encoder.config.projection_dim
        return (lambda out: out.image_embeds, {"return_loss": True if bool(self.args.encoder) and not bool(self.args.llm) else None})

    def build_hf_siglip(self, hf_encoder: AutoModel):
        VISION_ENCODERS[self.args.encoder]["model_hidden_size"] = hf_encoder.config.text_config.hidden_size
        VISION_ENCODERS[self.args.encoder]["projection_dim"] = hf_encoder.config.text_config.hidden_size
        return (lambda out: out.image_embeds, {"return_loss": True if bool(self.args.encoder) and not bool(self.args.llm) else None})

    def build_hf_vit(self, hf_encoder: AutoModel):
        VISION_ENCODERS[self.args.encoder]["projection_dim"] = hf_encoder.config.hidden_size
        VISION_ENCODERS[self.args.encoder]["model_hidden_size"] = hf_encoder.config.hidden_size
        VISION_ENCODERS[self.args.encoder]["num_patches"] = (hf_encoder.config.image_size // hf_encoder.config.patch_size) ** 2
        assert VISION_ENCODERS[self.args.encoder]["num_patches"] is not None, print("num_patches is None")

        return (
            lambda out: torch.mean(torch.mean(torch.stack(out.hidden_states), dim=0), dim=1),
            {"output_hidden_states": True if bool(self.args.encoder) and bool(self.args.llm) else None},
        )

    def load_encoder_checkpoint(self, encoder_components):
        encoder_checkpoint = torch.load(self.args.encoder_ckpt, map_location="cpu")
        encoder_config = VISION_ENCODERS if self.args.encoder in VISION_ENCODERS else ECG_ENCODERS
        strict_loading = encoder_config[self.args.encoder]["strict"]
        encoder_components["encoder"].load_state_dict(encoder_checkpoint["model_state_dict"], strict=strict_loading)
        if is_main():
            print(f"Loaded encoder checkpoint from {self.args.encoder_ckpt}")

    ### DEV FUNCTIONS ###
    def print_encoder_dtype(self, encoder):
        print("Encoder actual dtype:", next(encoder.parameters()).dtype)
