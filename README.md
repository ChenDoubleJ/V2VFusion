<div align="center">

# **V2VFusion: Text-Controlled Video-to-Video Diffusion for Degradation-Aware Video Fusion**

**A unified video-to-video fusion framework.**

<p>
  <img src="https://img.shields.io/badge/Conference-NeurIPS%202026-1f6feb" alt="Conference">
  <img src="https://img.shields.io/badge/Python-3.10-3776ab" alt="Python">
  <img src="https://img.shields.io/badge/Task-Video%20Fusion-6f42c1" alt="Task">
  <img src="https://img.shields.io/badge/Code-Released-2ea44f" alt="Code">
</p>

</div>

## **Updates**

- **2026.09.30**  The code has been *released*!
- **2026.09.25**  V2VFusion has been accepted by *NeurIPS* 2026!

## **Method Overview**

<p align="center">
  <img width="2218" height="831" alt="Method overview" src="https://github.com/user-attachments/assets/34a81992-e9f5-4fde-aa6a-b91d8ff02ed5" />
</p>

<p align="center">
  <img width="2087" height="1054" alt="Qualitative results" src="https://github.com/user-attachments/assets/1ff6af3a-81fd-4b6f-b0e8-6f0720fa05cd" />
</p>

## **Dependencies and Installation**

Clone this repository and create a conda environment:

```bash
git clone https://github.com/ChenDoubleJ/V2VFusion.git
cd V2VFusion

conda create -n v2vfusion python=3.10 -y
conda activate v2vfusion
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install the required system packages:

```bash
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6 -y
```

## **Inference**

### **Step 1: Prepare Model Weights**

Download the required checkpoints and place them under `pretrained_weight/`.

- Model weight: download from [model.safetensors](https://huggingface.co/Chendoublej/V2VFusion/resolve/main/model.safetensors?download=true).
- VAE: download from [stable-video-diffusion-img2vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid).
- CLIP-ViT text encoder: download from [CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K).

The expected layout is:

```text
pretrained_weight/
|-- model_weight/model.safetensors
|-- stable-video-diffusion-img2vid/
`-- CLIP-ViT-H-14-laion2B-s32B-b79K/
```



### **Step 2: Prepare Testing Data**

Put testing videos and text prompts under `data/`. V2VFusion supports three testing tasks: visible-infrared fusion, multi-exposure fusion, and multi-focus fusion. Each task uses the same folder structure:

```text
data/
|-- visible-infrared/
|   |-- lq/      # visible videos
|   |-- lq1/     # infrared videos
|   `-- text/    # text prompts
|-- multi-exposure/
|   |-- lq/      # over-exposure videos
|   |-- lq1/     # under-exposure videos
|   `-- text/    # text prompts
`-- multi-focus/
    |-- lq/      # near-focus videos
    |-- lq1/     # far-focus videos
    `-- text/    # text prompts
```

For each sample, put the paired videos in the corresponding `lq/` and `lq1/` folders, and put the matching text prompt in `text/`. The files for the same sample should use matching names across these three folders.

There are two ways to prepare text prompts:

1. Automatically generate prompts with a video-language model, such as [LLaVA-Video-7B-Qwen2](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2), or another suitable VL model.
2. Manually write prompts. *The visible video is degraded by..., the infrared video is degraded by ....*

For simple degradation tests, such as manually adding synthetic degradations, *writing prompts manually is usually the most convenient choice*. Model-generated prompts may require careful tuning and can consume additional computational resources.

### **Step 3: Run Inference**

Run the inference script:

```bash
bash video_fusion/scripts/inference.sh
```

Pay attention to the data and model weight paths:

```text
--input_path dataset/.../visible-infrared 

--model_path ./model.safetensors
```
