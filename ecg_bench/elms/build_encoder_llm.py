import argparse
from typing import Any, Dict, Iterable, Mapping

from ecg_bench.configs.constants import ECG_ENCODERS, VISION_ENCODERS


def merge_dicts(*parts: Mapping[str, Any], allow_override: Iterable[str] = ()) -> Dict[str, Any]:
    """Merge dict-like parts with duplicate-key protection.
    Keys in `allow_override` are allowed to be overwritten by later parts.
    Later parts win for allowed keys; duplicates for other keys raise."""
    out: Dict[str, Any] = {}
    allowed = set(allow_override)
    for p in parts:
        for k, v in p.items():
            if k in out and k not in allowed:
                raise ValueError(f"Duplicate component keys when merging: {k}")
            out[k] = v
    return out


class BuildEncoderLLM:
    def __init__(self, llm_components: dict, encoder_components: dict, args: argparse.Namespace):
        self.args = args
        self.llm_components = llm_components
        self.encoder_components = encoder_components

    def build_encoder_llm(
        self,
    ):
        if self.args.encoder == "projection":
            encoder_llm_components = self.build_fuyu()
        else:
            encoder_llm_components = self.build_llava()
        return merge_dicts(
            self.encoder_components,
            self.llm_components,
            encoder_llm_components,
            allow_override=("find_unused_parameters",),
        )

    def build_llava(
        self,
    ):
        from ecg_bench.elms.encoder_llm.llava import LLaVA
        from ecg_bench.elms.ecg_encoder.projection import Projection

        if self.args.encoder in VISION_ENCODERS:
            projection_dim = VISION_ENCODERS[self.args.encoder]["projection_dim"]
        else:
            projection_dim = ECG_ENCODERS[self.args.encoder]["projection_dim"]
        projection_layer = Projection(projection_dim, self.args.llm)
        encoder_llm = LLaVA(
            self.llm_components["llm"], self.encoder_components["encoder"], projection_layer, self.args.update_encoder, self.args.no_signal
        )
        return {"elm": encoder_llm}

    def build_fuyu(
        self,
    ):
        from ecg_bench.elms.encoder_llm.fuyu import Fuyu

        encoder_llm = Fuyu(self.llm_components["llm"], self.encoder_components["encoder"], self.args.no_signal)
        return {"elm": encoder_llm}
