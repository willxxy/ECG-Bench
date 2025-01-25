# ECG-Bench
Private repo for developing unified ECG generative benchmark

when cloning repo (For transformers sake) we work on 4.47.1

1. To install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain=1.79.0 -y`

2. Open a new terminal to set PATH for Rust installation.

3. After opening a new terminal, check the installation by running `rustc --version`.

4. Create the conda virtual environment via `conda create -n ecg-bench python=3.10.15`.

5. Activate the environment `conda activate ecg-bench`

6. `pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118`

7. `git clone https://github.com/willxxy/ECG-Bench.git`

8. `cd ECG-Bench`

9. `git submodule init`

10. `git submodule update`

11. Please `cd` into the `ECG-Bench/transformers` directory and `pip install -e .`.

12. Now `cd ../` and `cd` into the `ECG-Bench/ecg-plot` directory and `pip install -e .`.

13. Now `cd ../` and `pip install -e .`

14. Run the `ECG-Bench/test/test_gpu.py` to ensure you are able to use your GPU.

15. Run the `ECG-Bench/test/test_transformers.py` to ensure you properly installed the `transformers` package.

16. `cd` into `ECG-Bench/ecg_bench/rust_bpe` and execute `maturin develop --release` to compile the tokenizer.

17. Another consideration is that we use ***gated*** models (e.g., Llama 3.2, Gemma) from HuggingFace, therefore you will need to get an api key and log into it via `huggingface-cli login` in the terminal. We also require you to log in inside the main training *.py file via the login function `from huggingface_hub import login`.

#
Run tests in ECG-Bench
example: python tests/test_file.py


## Checklist
[x] Preprocessing (preprocessing, percentiles, sampling)

[ ] Preprocessing tests

[X] Visualizer (plot 1d, 2d, other things) 

[ ] Visualizer tests

[ X] Tokenizer (bpe for now)

[X ] Tokenizer tests

[ X] main pipeline
    - [ X] .py files for pretraining and finetuning

[X ] models (llm, encoders, etc.)

[ X] runners
    - [ X] .py files for training and evaluating

[ ]  Add ECG instruct and datasets from PULSE paper.
