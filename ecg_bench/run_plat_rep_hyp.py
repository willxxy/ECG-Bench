import gc
import torch

torch.set_num_threads(6)
from ecg_bench.configs.config import get_args
from ecg_bench.utils.gpu_setup import GPUSetup
from ecg_bench.utils.set_seed import set_seed
from ecg_bench.dataloaders.build_dataloader import BuildDataLoader
from ecg_bench.elms.build_elm import BuildELM
from ecg_bench.runners.plat_rep_hyp import run_plat_rep_hyp_sep, run_plat_rep_hyp_comb


def main():
    gc.collect()
    torch.cuda.empty_cache()
    mode = "train"
    args = get_args(mode)
    set_seed(args.seed)
    build_dataloader = BuildDataLoader(mode, args)
    dataloader = build_dataloader.build_dataloader()
    build_elm = BuildELM(args)
    elm_components = build_elm.build_elm(build_dataloader.llm_tokenizer_components["llm_tokenizer"])
    gpu_setup = GPUSetup(args)
    elm = gpu_setup.setup_gpu(elm_components["elm"], elm_components["find_unused_parameters"])
    if args.dev:
        gpu_setup.print_model_device(elm, f"{args.llm}_{args.encoder}")
    if args.plat_rep_type == "separate":
        run_plat_rep_hyp_sep(elm, dataloader, args)
    elif args.plat_rep_type == "combined":
        run_plat_rep_hyp_comb(elm, dataloader, args)


if __name__ == "__main__":
    main()
