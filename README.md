<!--# [SegChange-R1: LLM-Augmented Remote Sensing Change Detection](https://arxiv.org/abs/2506.17944) -->

English | [简体中文](README_zh.md) | [English](README.md) | [CSDN Blog](https://blog.csdn.net/weixin_62828995?spm=1000.2115.3001.5343)

<h2 align="center">
  SegChange-R1: LLM-Augmented Remote Sensing Change Detection
</h2>

<p align="center">
    <a href="https://huggingface.co/spaces/yourusername/TAPNet">
        <img alt="hf" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
    </a>
    <a href="https://github.com/Yu-Zhouz/SegChange-R1/blob/master/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a>
    <a href="https://github.com/Yu-Zhouz/SegChange-R1/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/Yu-Zhouz/SegChange-R1">
    </a>
    <a href="https://github.com/Yu-Zhouz/SegChange-R1/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/Yu-Zhouz/SegChange-R1?color=olive">
    </a>
    <a href="https://arxiv.org/abs/2506.17944">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2505.06937v1-red">
    </a>
    <a href="https://results.pre-commit.ci/latest/github/Yu-Zhouz/SegChange-R1/master">
        <img alt="pre-commit.ci status" src="https://results.pre-commit.ci/badge/github/Yu-Zhouz/SegChange-R1/master.svg">
    </a>
    <a href="https://github.com/Yu-Zhouz/SegChange-R1">
        <img alt="stars" src="https://img.shields.io/github/stars/Yu-Zhouz/SegChange-R1">
    </a>
</p>

<p align="center">
    📄 This is the official implementation of the paper:
    <br>
    <a href="https://arxiv.org/abs/2506.17944">SegChange-R1: LLM-Augmented Remote Sensing Change Detection</a>
</p>

<p align="center">
Fei Zhou
</p>

<p align="center">
Neusoft Institute Guangdong, China & Airace Technology Co.,Ltd., China
</p>

<p align="center">
<strong>If you like SegChange-R1, please give us a ⭐! </strong>
</p>

Remote sensing change detection is used in urban planning, terrain analysis, and environmental monitoring by analyzing feature changes in the same area over time. In this paper, we propose a large language model (LLM) augmented inference approach (SegChange-R1), which enhances the detection capability by integrating textual descriptive information and guides the model to focus on relevant change regions, accelerating convergence. We designed a linear attention-based spatial transformation module (BEV) to address modal misalignment by unifying features from different times into a BEV space. Furthermore, we introduce DVCD, a novel dataset for building change detection from UAV viewpoints. Experiments on four widely-used datasets demonstrate significant improvements over existing method. 

![Baseline](https://i-blog.csdnimg.cn/direct/574c18c2382c442a8cb60b31de9c01ba.png)

## 🚀 Updates

- ✅ **[2024.06.01]** Open source code
- ✅ **[2025.06.22]** Upload to [arXiv](https://arxiv.org/abs/2506.17944)。

## Model Zoo

![SOTA](https://i-blog.csdnimg.cn/direct/186edd2273c149bcb19f9aafb60b835b.png)


## Quick Start

### System Requirements

- Python 3.12
- CUDA + PyTorch
- HuggingFace
- Stable network connection
- High-quality proxy IPs (important)

### Installation

#### 1. Create a virtual environment

```bash
conda create -n segchange python=3.12 -y
conda activate segchange
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Configuring HuggingFace Images

```bash
vim ~/.bashrc

export HF_ENDPOINT="https://hf-mirror.com"

source ~/.bashrc
```

### Data Preparation

Two dataset structure formats are supported：

#### 1. Default

The structure of the dataset is as follows:

```text
data/
  ├── train/
  │   ├── A/                  # First phase training image
  │   ├── B/                  # Second phase training image
  │   ├── label/              # Training Label (Change Mask)
  │   └── prompts.txt         # The training set text describes the prompt
  ├── val/
  │   ├── A/              
  │   ├── B/             
  │   ├── label/         
  │   └── prompts.txt 
  └── test/
      ├── A/            
      ├── B/             
      ├── label/      
      └── prompts.txt     
```
Change the `data_format` parameter file [configs](./configs/config.yaml) to `default`.

#### 2. Custom

The structure of the dataset is as follows:

```text
data/
  ├── A/                      # First phase training image
  │── B/                      # Second phase training image
  │── label/                  # Label (Change Mask)
  │── list                    # List file
  │   ├── train.txt           # A list of training sets
  │   ├── val.txt             # A list of validation sets
  │   └── test.txt            # A list of test sets
  └── prompts.txt             # Text description prompts
```
Change the `data_format` parameter file [configs](./configs/config.yaml) to `custom`.

### Training

#### Generate word embedding files

Use [Text Generation](./examples/text_gen.py) and change the 'desc_embs' parameter file [configs](./configs/config.yaml) to 'None' to execute the script.

```bash
python ./examples/text_gen.py -c ./configs/config.yaml
```
If you want to detect changes in multiple categories, you need to manually label the category description text.

#### Command-line training
```bash
python train.py -c ./configs/config.yaml
```

### testing
```bash
python test.py -c ./configs/config.yaml
```

### Inference TIF
```bash
python infer.py -c ./configs/config.yaml
```

### app demo

```bash
cd examples/gradio_app
chmod +x ./run.sh
bash run.sh
```
### DEDICATION

Submit issues and code improvements. Make sure to follow the project's code style and contribution guidelines.

## LICENSE

This project uses [Apache License 2.0](LICENSE)

### Citation
If you use RT-FINE in your research, please cite:

<details open>
<summary> bibtex </summary>

```bibtex
@article{zhou2025segchange-r1,
  title={SegChange-R1: LLM-Augmented Remote Sensing Change Detection},
  author={Zhou, Fei},
  journal={arXiv preprint arXiv:2506.17944},
  year={2025},
  eprint={/2506.17944},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
</details>
---

# RTS Change Detection Ablation (this branch)

Ablation over how much semantic/text guidance helps Arctic retrogressive thaw slump (RTS)
change detection, on top of upstream
[SegChange-R1](https://arxiv.org/abs/2506.17944). See the top-level README for
install/environment setup (conda env `segchange`).

## Experiments

Four configs in `configs/`, each with its own `data.data_root` and `work_dirs` name:

| config | text input | text encoder | frozen |
|---|---|---|---|
| `config_exp1_notext.yaml` | none | bert-base-uncased (unused) | — |
| `config_exp2_static.yaml` | same fixed phrase for every image | bert-base-uncased | yes |
| `config_exp3_dynamic.yaml` | real per-image description | bert-base-uncased | yes |
| `config_exp4_llm.yaml` | same per-image description as exp3 | microsoft/phi-1_5 | yes |

exp1 vs exp2: does text presence alone help, even with zero per-image information.
exp2 vs exp3: does per-image-accurate text help over a generic fixed phrase.
exp3 vs exp4: does a larger causal-LM text encoder beat BERT, holding the text and the
frozen/not-fine-tuned setting fixed (both encoders frozen, so this isolates encoder choice).

## Data layout

Each `data/change/exp{1,2,3,4}_*/` needs:
```
train/{A,B,label}/   # image pairs + change mask, same filenames across A/B/label
val/{A,B,label}/
train/prompts.txt    # absent for exp1_notext
val/prompts.txt
```

`prompts.txt`: one line per image, **double-space separated**:
```
<filename>.png  <description text>
```
`exp2_static`: every line has the same generic phrase (carries no per-image signal, by
design). `exp3_dynamic` / `exp4_llm`: real per-image descriptions, identical between the two
(same text, different downstream encoder). Negative (no-change) samples use a fixed phrase:
`"No retrogressive thaw slumps / No change in the retrogressive thaw slumps area."`

## Running

```bash
python train.py -c configs/config_exp1_notext.yaml
python train.py -c configs/config_exp2_static.yaml
python train.py -c configs/config_exp3_dynamic.yaml
python train.py -c configs/config_exp4_llm.yaml
```

`config_exp4_llm.yaml` uses `microsoft/phi-1_5`; set `HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1` if the weights are already cached locally, to avoid a network stall
on startup.
