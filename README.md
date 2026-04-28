# ThoraxAI-Equity: Few-Shot ICL and Dual-Agent Ensemble for Pneumothorax Detection

Code repository for the paper:

**"Comprehensive Evaluation of Few-Shot In-Context Learning and Dual-Agent Ensemble Strategies for Pneumothorax Detection Using Resource-Efficient Vision-Language Models"**

## Overview

This repository contains all experiment code for evaluating four resource-efficient VLMs (3-4B parameters, 4-bit NF4 quantized) on pneumothorax detection from chest radiographs using few-shot in-context learning (ICL) and dual-agent ensemble strategies.

## Models

All models are 3-4B parameters and quantized to 4-bit NF4, designed to run on minimal hardware (single consumer GPU) for practical clinical deployment.

| ID | Model | Parameters | Training Strategy |
|----|-------|-----------|-------------------|
| M1 | [NV-Reason-CXR-3B](https://github.com/NVIDIA-Medtech/NV-Reason-CXR) | 3B | CXR-reasoning fine-tune |
| M2 | [ChestXReasoner](https://github.com/MAGIC-AI4Med/ChestX-Reasoner) | 3B | CXR-direct fine-tune |
| M3 | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF) | 3B | No medical fine-tune |
| M4 | [MedGemma-4B-IT](https://huggingface.co/google/medgemma-4b-it) | 4B | Medical-general fine-tune |

## Repository Structure

```
├── src/                           # Core source modules
│   ├── inference.py               # VLM inference pipeline
│   ├── prompt_builder.py          # Prompt construction (system, user, few-shot)
│   ├── example_selector.py        # Few-shot example sampling
│   ├── evaluate.py                # Metrics computation
│   └── run_experiments.py         # Experiment runner
│
├── stage-1/                       # Stage 1: Individual ICL inference
│   ├── M1.ipynb                   # NV-Reason-CXR-3B (k=0,1,2,4,8)
│   ├── M2.ipynb                   # ChestXReasoner
│   ├── M3.ipynb                   # Qwen2.5-VL-3B-Instruct
│   └── M4.ipynb                   # MedGemma-4B-IT
│
├── stage-2/                       # Stage 2: Analysis & bias profiling
│   ├── analyze-M1/                # M1 analysis (10 seeds)
│   │   ├── 3-analysis-exp1.ipynb
│   │   ├── figures-exp-1/         # Generated figures (png/pdf)
│   │   └── results-exp-1/         # Raw results (jsonl)
│   ├── analyze-M2/                # M2 analysis
│   ├── analyze-M3/                # M3 analysis
│   └── analyze-M4/                # M4 analysis
│
├── stage-3/                       # Stage 3: Ensemble experiments
│   ├── after-inference/           # Scenario A (S1-S5 strategies)
│   │   ├── 5-ensemble-p1.ipynb
│   │   └── 5-ensemble-p1-m1-m4.ipynb
│   └── during-inference/          # Scenario C (D1-D4 strategies)
│       ├── 6-during-ensemble-p1-m1-m4-done.ipynb
│       ├── 7-...-reasoning-1.ipynb    # Prompt v1 (biased)
│       ├── 7-...-reasoning-2.ipynb    # Prompt v2 (balanced)
│       ├── 7-...-reasoning-3-confidence.ipynb  # Prompt v3 (confidence)
│       └── 8-reversed-during-ensemble-m4-m1-confidence.ipynb
│
└── requirements.txt
```

## Datasets

Datasets are available on Kaggle: **[Pneumothorax Chest X-ray Indonesia](https://www.kaggle.com/datasets/artakusuma/pneumothorax-chest-x-ray-indonesia)**

- **[CheXpert Plus](https://aimi.stanford.edu/datasets/chexpert-plus)** (Stanford) -- Few-shot pool (300 images) + Internal test (200 images)
- **RS UNAIR** (Universitas Airlangga Hospital, Surabaya) -- External test (278 images)

## Setup

```bash
pip install -r requirements.txt
```

## Hardware

- **CPU:** 13th Gen Intel Core i9-13900KF
- **GPU:** NVIDIA RTX 5090
- **RAM:** 64 GB
- **OS:** Ubuntu 24.04

## Key Results

| Approach | Strategy | F1 (Internal) | F1 (External) |
|----------|----------|---------------|---------------|
| Single | M1 best (k=8) | 0.670 | 0.731 |
| Single | M4 best (k=8) | 0.531 | 0.549 |
| After-Inference | S-A S2 OR-gate (k=4) | **0.683** | **0.750** |
| During-Inference | S-C v3 D4 (k=8) | 0.672 | 0.738 |

## Citation

```
[To be added after publication]
```

## Acknowledgments

This work was supported by the Indonesia Endowment Fund for Education (LPDP) through the EQUITY-WCU program (ITS-LPDP-Kemendiktisaintek).
