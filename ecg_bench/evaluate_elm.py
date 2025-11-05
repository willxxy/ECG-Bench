import gc
import json
import os
import torch

torch.set_num_threads(6)
from ecg_bench.configs.config import get_args
from ecg_bench.utils.gpu_setup import GPUSetup
from ecg_bench.utils.set_seed import set_seed
from ecg_bench.dataloaders.build_dataloader import BuildDataLoader
from ecg_bench.elms.build_elm import BuildELM
from ecg_bench.runners.elm_evaluator import evaluate, run_statistical_analysis


def main():
    gc.collect()
    torch.cuda.empty_cache()
    mode = "eval"
    args = get_args(mode)
    # folds = ["1", "2", "3", "4", "5"]
    # seeds = [1337, 1338, 1339, 1340, 1341]
    folds = ["1", "2"]
    seeds = [1337, 1338]
    all_metrics = []
    for fold, seed in zip(folds, seeds):
        print(f"Evaluating fold {fold} with seed {seed}")
        args.fold = fold
        args.seed = seed
        set_seed(args.seed)
        build_dataloader = BuildDataLoader(mode, args)
        dataloader = build_dataloader.build_dataloader()
        build_elm = BuildELM(args)
        elm_components = build_elm.build_elm(build_dataloader.llm_tokenizer_components["llm_tokenizer"])
        gpu_setup = GPUSetup(args)
        elm = gpu_setup.setup_gpu(elm_components["elm"], elm_components["find_unused_parameters"])
        if args.dev:
            gpu_setup.print_model_device(elm, f"{args.llm}_{args.encoder}")
        out = evaluate(elm, dataloader, args)
        all_metrics.append(out)

    statistical_results = run_statistical_analysis(all_metrics)
    print(statistical_results)

    checkpoint_dir = os.path.dirname(args.elm_ckpt)
    results_file = os.path.join(checkpoint_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(statistical_results, f, indent=2)
    print(f"Saved evaluation results to {results_file}")


if __name__ == "__main__":
    main()
