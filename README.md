<h2 align="center">
  Time Series, Image, or Text: Exploring the Best Input Representation for Electrocardiogram-Language Models Through a Unified Framework
</h2>

<div align="center">
  <img src="./assets/fig1_2.png" alt="Our pipeline.">
</div>

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [ECG Datasets](#data)
4. [Main Methods](#methods)
  - [Training ECG-Byte](#ecg-byte)
  - [Training LLM](#endtoend-train)
  - [Inference LLM](#endtoend-inference)
  - [Analysis of methods](#endtoend-analysis)
  - [Training 1st Stage](#twostage-train1)
  - [Training 2nd Stage](#twostage-train2)
  - [Inferencing 2 Stage](#twostage-inference)
5. [Known Issues + Tips](#issues)
6. [Acknowledgements](#ack)


## Overview <a name="overview"></a>
This repository is the official implementation of [Time Series, Image, or Text: Exploring the Best Input Representation for Electrocardiogram-Language Models Through a Unified Framework]()
by [William Jongwon Han](https://willxxy.github.io/), [Choajing Duan](https://www.linkedin.com/in/chaojing-duan-0b3266127), [Michael A. Rosenberg](https://scholar.google.com/citations?user=o0Y0GLcAAAAJ&hl=en), [Emerson Liu](https://www.linkedin.com/in/emerson-liu-950479/), and [Ding Zhao](https://www.meche.engineering.cmu.edu/directory/bios/zhao-ding.html).

Although this repository is based on the mentioned paper, the overall goal of the repository is to provide a unified framework for training and evaluating electrocardiogram-language models (ELMs).
To this end, we provide the following features not mentioned in the paper:
1. We impemented an LLM judge with llm-blender and utilized [DPO](https://arxiv.org/abs/2305.18290) for post-training.
2. Flash Attention 2 for faster training and inference.

We hope to continouously update the repository to support more features, ELMs, and datasets. Please feel free to contribute to the repository!
Please carefully read the below documentations to run the pipeline. If there are any questions or bugs, please do not hesitate to reach out to wjhan{@}andrew{dot}cmu{edu} or submit an issue with corresponding details.

All installations and experiments were completed on Ubuntu 20.04.5 LTS with NVIDIA A6000 GPUs.

## Installation <a name="installation"></a>

1. To install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain=1.79.0 -y`

2. Open a new terminal to set PATH for Rust installation.

3. After opening a new terminal, check the Rust installation by running `rustc --version`.

4. Create the conda virtual environment via `conda create -n elm python=3.10.15`.

5. Activate the environment `conda activate elm`

6. `pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121`

7. `git clone https://github.com/willxxy/ELM-Bench.git`

8. `cd ELM-Bench`

9. `git submodule init`

10. `git submodule update`

11. Please `cd` into the `ELM-Bench/transformers` directory and `pip install -e .`.

12. Now `cd ../` and `cd` into the `ELM-Bench/ecg-plot` directory and `pip install -e .`.

13. Now `cd ../` and `pip install -e .`

14. We do not use [Flash Attention 2](https://arxiv.org/abs/2307.08691) for the results in the paper but we provide the option to install flash-attn to speed up training and inference, however, it is not required. To install flash-attn please use the following command:

    `pip cache remove flash_attn`

    `pip install flash-attn==2.7.4.post1 --no-cache-dir`

15. Install the `llm-blender` and `trl[judges]` packagesby running the following commands:

    `pip install git+https://github.com/yuchenlin/LLM-Blender.git`

    `pip install trl[judges]`

16. Run the `ELM-Bench/test/test_gpu.py` to ensure you are able to use your GPU.

17. Run the `ELM-Bench/test/test_transformers.py` to ensure you properly installed the `transformers` package.

18. `cd` into `ELM-Bench/elm_bench/rust_bpe` and execute `maturin develop --release` to compile the tokenizer.

19. Another consideration is that we use ***gated*** models (e.g., Llama 3.2, Gemma) from HuggingFace, therefore you will need to get an api key and log into it via `huggingface-cli login` in the terminal. We also require you to log in inside the main training *.py file via the login function `from huggingface_hub import login`.


**NOTE: From now, all instructions will assume you are working from the `ELM-Bench/elm_bench` directory.**

## ECG Datasets <a name="data"></a>

### Base Datasets

We regard base datasets as datasets that are solely used for later mapping of external datasets.

#### PTB-XL

1. Please download the PTB-XL dataset through this [link](https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip).

2. Please create a `data` folder, unzip the zip file inside the `data` folder and rename the folder as `ptb`.

#### MIMIC

1. Please download the Mimic IV ECG dataset through this [link](https://physionet.org/static/published-projects/mimic-iv-ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip).

2. Unzip the zip file inside the `data` directory and rename the unzipped directory as `mimic`.

#### Code-15

1. First create a `code15` folder inside the `data` directory.

2. Then inside `data/code15` execute the following bash script to download the data and unzip it:

```
#!/bin/bash

for i in {0..17}; do
    echo "Downloading part ${i}..."
    wget -O "exams_part${i}.zip" "https://zenodo.org/records/4916206/files/exams_part${i}.zip?download=1"
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded part ${i}"
        
        echo "Extracting part ${i}..."
        unzip -q "exams_part${i}.zip"
        
        if [ $? -eq 0 ]; then
            echo "Successfully extracted part ${i}"
            rm "exams_part${i}.zip"
        else
            echo "Error extracting part ${i}"
        fi
    else
        echo "Error downloading part ${i}"
    fi
done

echo "All downloads and extractions completed"
```

#### CSN

1. Create a `csn` folder inside the `data` directory.

2. Inside `data/csn` execute the following command in the terminal:

```
wget https://physionet.org/static/published-projects/ecg-arrhythmia/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip
```

3. Unzip the file and inside of `data/csn/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0` move all of the contents outside to `data/csn`. Then you may delete the `a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0` folder.

#### CPSC

1. Create a `cpsc` folder inside the `data` directory.

2. Inside `data/cpsc` execute the following command in the terminal:

```
wget https://physionet.org/static/published-projects/challenge-2020/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2.zip
```

3. Unzip the file and inside of `data/cpsc/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2/training` move the `cpsc_2018` and `cpsc_2018_extra` folders into the `data/cpsc` directory. Then delete the `classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2` folder.

### External Datasets

External datasets are datasets that are mapped to the base datasets and subsequently used for all experiments.

#### ECG-QA dataset curated by [ECG-QA, Oh et al.](https://github.com/Jwoo5/ecg-qa)

1. To download the ECG-QA dataset, please execute the following command in the `data` folder:

`git clone https://github.com/Jwoo5/ecg-qa.git`

2. We exactly follow the instructions in [this section of the repository](https://github.com/Jwoo5/ecg-qa?tab=readme-ov-file#usage-notes) for mapping the PTB-XL and MIMIC IV ECG dataset to the question and answers. `cd` into ecg-qa and execute the following commands in the terminal to prepare the ECG-QA dataset.

3. To map the ECG-QA dataset to mimic and ptb `cd` inside the `data/ecg-qa` directory and execute the following scripts respectively.

```
python mapping_ptbxl_samples.py ecgqa/ptbxl \
--ptbxl-data-dir ../ptb
```

```
python mapping_mimic_iv_ecg_samples.py ecgqa/mimic-iv-ecg \
--mimic-iv-ecg-data-dir ../mimic
```

3. After mapping the datasets, you should have an output folder in the `data/ecg-qa` folder with the mapped `paraphrased` and `template` question and answers.

#### Pretrain MIMIC dataset curated by [ECG-Chat, Zhao et al.](https://github.com/YubaoZhao/ECG-Chat)

1. Next create a `data/pretrain_mimic` directory and download the `pretrain_mimic.json` file from this [dropbox link](https://www.dropbox.com/scl/fo/ccq5dxmdgg4shf02yjn8c/ANOQ1Hzj4KwHqa1b9r80uzc?rlkey=teysp3v6hg6o9uko2i4zbbjpn&e=1&st=exu3i9oo&dl=0).

### Instruct 45k MIMIC dataset curated by [ECG-Chat, Zhao et al.](https://github.com/YubaoZhao/ECG-Chat)

1. Next create a `data/ecg_instruct_45k` directory and download the `ecg_instruct_45k.json` file from this [link](https://github.com/YubaoZhao/ECG-Chat/blob/master/llava/playground/data/ecg_instruct_45k.json).


#### ECG Instruct Pulse dataset curated by [PULSE, Liu et al.](https://github.com/AIMedLab/PULSE)

1. Create a 'data/ecg_instruct_pulse' directory and downlod the `ECGInstruct.json`from this [link](https://huggingface.co/datasets/PULSE-ECG/ECGInstruct/tree/main). Then rename it to `ecg_instruct_pulse.json`.

Once you are finished with these steps, it's time to preprocess the data!

### Preprocessing

1. Execute the preprocessing script by `bash scripts/preprocess.sh`. We have provided default configurations for all the datasets used in our study but feel free to experiment with others!


## Main Methods <a name="methods"></a>

### Training ECG-Byte <a name="ecg-byte"></a>

1. Once you sampled the ECGs, you can simply run `bash scripts/train_tokenizer.sh` to train the tokenizer. We also provide a script to load in your trained tokenizer and see the encoding compression rate and original vs. decoded signal. Lastly, we provide basic configurations, however, please feel free to modify these.

NOTE: We also provide a trained tokenizer at this [link](https://drive.google.com/drive/folders/1IFrg-XRRDhJ_xIUSxjcXwxvyPLdsbMR0?usp=sharing). Please feel free to use this or train your own!


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
