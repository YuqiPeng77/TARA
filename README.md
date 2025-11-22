# TARA: Token-Aware LoRA for Composable Personalization in Diffusion Models

This is the official implementation of **[TARA: Token-Aware LoRA for Composable Personalization in Diffusion Models](https://arxiv.org/abs/2508.08812)**.

TARA introduces a simple yet effective framework for composable personalization by addressing token interference and spatial misalignment in multi-LoRA generation.

> 🔍 TARA allows **multiple LoRA modules** to be trained independently and composed at inference time **without extra merging or joint fine-tuning**.
> 

## 📰 News

* **2025.11 — Our paper *TARA: Token-Aware LoRA for Composable Personalization in Diffusion Models* has been accepted to AAAI 2026! 🎉**


## 🧠 Highlights

- **Token Focus Masking (TFM):** Ensures each LoRA is only active for its target token.
- **Token Alignment Loss (TAL):** Guides each token to align with its corresponding visual region.
- **Composable Inference:** Multiple LoRAs can be used together during generation without retraining.


## 📋 Installation

```bash
conda env create -f environment.yml
conda activate tara
```

Make sure you have a GPU environment with compatible CUDA and PyTorch versions.

## 📂 Dataset

We use the [DreamBooth dataset](https://github.com/google/dreambooth) to train personalized LoRA modules.  

Each instance contains images of a single subject along with a prompt describing its class and identifier (e.g., `a photo of token1 person`).

Please follow the instructions in the DreamBooth repository to prepare your dataset in the correct format.

## 🏋️‍♂️ Training

You can train LoRA modules for either Stable Diffusion 1.5 or SDXL. Choose the appropriate script and base model as shown below.

### Train on Stable Diffusion 1.5

To train a LoRA module on Stable Diffusion 1.5 for a single concept:

```bash
python train_dreambooth_tara_sd.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-v1-5" \
    --instance_data_dir="path/to/your/images" \
    --instance_prompt="a photo of <rare_token> <class_word>" \
    --output_dir="path/to/save/lora" \
    --use_lora \
    --train_batch_size=1 \
    --max_train_steps=1000 \
    --learning_rate=1e-4 \
    --resolution=512 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --rare_word="<rare_token>" \
    --class_word="<class_word>" \
    --focus true \
    --with_align_loss true \
    --align_loss_weight=1.0
```

### Train on Stable Diffusion XL

To train a LoRA module on SDXL for a single concept:

```bash
train_dreambooth_tara_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --instance_data_dir="path/to/your/images" \
    --output_dir="path/to/save/lora" \
    --instance_prompt="a photo of <rare_token> <class_word>" \
    --rank=8 \
    --resolution=1024 \
    --train_batch_size=1 \
    --learning_rate=5e-5 \
    --report_to="wandb" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --seed=$SEED \
    --mixed_precision="fp16" \
    --gradient_checkpointing \
    --use_8bit_adam \
    --rare_word="<rare_token>" \
    --class_word="<class_word>" \
    --focus="true"  \
    --with_align_loss="true"  \
    --align_loss_weight=1.0 \
```

📌 Replace:
- `<rare_token>` with your unique concept identifier (e.g., `xlo`).
- `<class_word>` with a general category word (e.g., `dog`).


## 🖼️ Inference

Inference with single LoRA module:

```bash
python inference_tara.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-v1-5" \
  --lora_list='["path/to/lora.safetensors"]' \
  --rare_token_list='["token"]' \
  --prompt="A <rare_token> <class_word>." \
  --output_folder="./output" \
  --number=4
```

Inference with multiple LoRA modules:

```bash
python inference_tara.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-v1-5" \
  --lora_list='["path/to/lora1.safetensors", "path/to/lora2.safetensors", ...]' \
  --rare_token_list='["token1", "token2", ...]' \
  --prompt="A <rare_token1> <class_word1> and a <rare_token12> <class_word12> ..." \
  --output_folder="./output" \
  --number=4
```



## 📁 Folder Structure

```
.
├── train_dreambooth_tara_sd.py      # Training script for SD V1.5
├── train_dreambooth_tara_sdxl.py    # Training script for SDXL V1.0
├── inference_tara.py                # Inference script
├── environment.yml                  # Conda environment
├── utils.py / tara.py               # Utility and core logic
```


## 📌 Notes
- Based on 🤗 HuggingFace Diffusers, supporting both Stable Diffusion v1.5 and SDXL.
- Supports LoRA modules trained independently and used flexibly at inference.

