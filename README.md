# ECG-Bench
Private repo for developing unified ECG generative benchmark

when cloning repo (For transformers sake) we work on 4.47.1

1. git clone repo
2. cd ECG-Bench
3. git submodule init
4. git submodule update

#
Run tests in ECG-Bench
example: python tests/test_file.py


## Checklist
[x] Preprocessing (preprocessing, percentiles, sampling)
[ ] Preprocessing tests
[ ] Visualizer (plot 1d, 2d, other things) 
[ ] Visualizer tests
[ ] Tokenizer (bpe for now)
[ ] Tokenizer tests
[ ] main pipeline
    - [ ] .py files for pretraining and finetuning
[ ] models (llm, encoders, etc.)
[ ] runners
    - [ ] .py files for training and evaluating
[ ] 