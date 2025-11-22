                                               
import argparse
import gc
import hashlib
import itertools
import logging
import math
import os
import threading
import warnings
from contextlib import nullcontext
from pathlib import Path

import datasets
import diffusers
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfApi
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from peft import LoraConfig, get_peft_model
from tara import TARALinearLayer
from diffusers.models.lora import LoRALinearLayer, LoRACompatibleLinear


                                                                                            
check_min_version("0.10.0.dev0")

logger = get_logger(__name__)

UNET_TARGET_MODULES = ["to_k", "to_out.0", "to_q", "to_v", "query", "value"]
TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--focus",
        type=str2bool,
        default="false",
        help="add token focus loss.",
    )
    parser.add_argument(
        "--rare_word",
        type=str,
        default="",
        help="The rare word to be used for cross attention regularization.",
    )
    parser.add_argument(
        "--class_word",
        type=str,
        default="",
        help="The class word to be used for cross attention regularization.",
    )
    parser.add_argument(
        "--with_align_loss",
        type=str2bool,
        default="false",
        help="add token alignment loss.",
    )
    parser.add_argument(
        "--align_loss_weight",
        type=float,
        default=1.0,
        help="weight for token alignment loss.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run dreambooth validation every X steps. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")

               
    parser.add_argument("--use_lora", action="store_true", help="Whether to use Lora for parameter efficient tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="Lora rank, only used if use_lora is True")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Lora alpha, only used if use_lora is True")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Lora dropout, only used if use_lora is True")
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if use_lora is True",
    )
    parser.add_argument(
        "--lora_text_encoder_r",
        type=int,
        default=8,
        help="Lora rank for text encoder, only used if `use_lora` and `train_text_encoder` are True",
    )
    parser.add_argument(
        "--lora_text_encoder_alpha",
        type=int,
        default=32,
        help="Lora alpha for text encoder, only used if `use_lora` and `train_text_encoder` are True",
    )
    parser.add_argument(
        "--lora_text_encoder_dropout",
        type=float,
        default=0.0,
        help="Lora dropout for text encoder, only used if `use_lora` and `train_text_encoder` are True",
    )
    parser.add_argument(
        "--lora_text_encoder_bias",
        type=str,
        default="none",
        help="Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if use_lora and `train_text_encoder` are True",
    )

    parser.add_argument(
        "--num_dataloader_workers", type=int, default=1, help="Num of workers for the training dataloader."
    )

    parser.add_argument(
        "--no_tracemalloc",
        default=False,
        action="store_true",
        help="Flag to stop memory allocation tracing during training. This could speed up training on Windows.",
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        default=None,
        help=("If report to option is set to wandb, api-key for wandb used for login to wandb "),
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help=("If report to option is set to wandb, project name in wandb for log tracking  "),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
                                     
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


                               
def b2mb(x):
    return int(x / 2**20)


class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()                                
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)
            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )
    if args.report_to == "wandb":
        import wandb

        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project_name)
                                                                                                                  
                                                                                                                      
                                                                                                                
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

                                                                         
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

                                                 
    if args.seed is not None:
        set_seed(args.seed)

                                                             
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

                                    
    if accelerator.is_main_process:
        if args.push_to_hub:
            api = HfApi(token=args.hub_token)

                                                           
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

                        
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

                                       
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

                               
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )                                                                                            
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    if args.use_lora:
        if args.focus:
            unet_lora_parameters = []                                                         
            for attn_processor_name, attn_processor in unet.attn_processors.items():
                is_cross = ".attn2" in attn_processor_name
                                             
                attn_module = unet
                for n in attn_processor_name.split(".")[:-1]:
                    attn_module = getattr(attn_module, n)

                                                                                            
                      
                old_layer = attn_module.to_q
                new_layer = LoRACompatibleLinear(
                    in_features=old_layer.in_features,
                    out_features=old_layer.out_features,
                    bias=old_layer.bias is not None,
                    device=old_layer.weight.device,
                    dtype=old_layer.weight.dtype,
                )
                new_layer.weight.data.copy_(old_layer.weight.data)
                if old_layer.bias is not None:
                    new_layer.bias.data.copy_(old_layer.bias.data)
                attn_module.to_q = new_layer

                      
                old_layer = attn_module.to_k
                new_layer = LoRACompatibleLinear(
                    in_features=old_layer.in_features,
                    out_features=old_layer.out_features,
                    bias=old_layer.bias is not None,
                    device=old_layer.weight.device,
                    dtype=old_layer.weight.dtype,
                )
                new_layer.weight.data.copy_(old_layer.weight.data)
                if old_layer.bias is not None:
                    new_layer.bias.data.copy_(old_layer.bias.data)
                attn_module.to_k = new_layer

                      
                old_layer = attn_module.to_v
                new_layer = LoRACompatibleLinear(
                    in_features=old_layer.in_features,
                    out_features=old_layer.out_features,
                    bias=old_layer.bias is not None,
                    device=old_layer.weight.device,
                    dtype=old_layer.weight.dtype,
                )
                new_layer.weight.data.copy_(old_layer.weight.data)
                if old_layer.bias is not None:
                    new_layer.bias.data.copy_(old_layer.bias.data)
                attn_module.to_v = new_layer

                           
                old_layer = attn_module.to_out[0]
                new_layer = LoRACompatibleLinear(
                    in_features=old_layer.in_features,
                    out_features=old_layer.out_features,
                    bias=old_layer.bias is not None,
                    device=old_layer.weight.device,
                    dtype=old_layer.weight.dtype,
                )
                new_layer.weight.data.copy_(old_layer.weight.data)
                if old_layer.bias is not None:
                    new_layer.bias.data.copy_(old_layer.bias.data)
                attn_module.to_out[0] = new_layer

                                         
                attn_module.to_q.set_lora_layer(
                    TARALinearLayer(
                        in_features=attn_module.to_q.in_features,
                        out_features=attn_module.to_q.out_features,
                        rank=args.lora_r,
                        network_alpha=args.lora_alpha,
                        use_mask=False,
                    )
                )
                attn_module.to_k.set_lora_layer(
                    TARALinearLayer(
                        in_features=attn_module.to_k.in_features,
                        out_features=attn_module.to_k.out_features,
                        rank=args.lora_r,
                        network_alpha=args.lora_alpha,
                        use_mask=is_cross,
                    )
                )
                attn_module.to_v.set_lora_layer(
                    TARALinearLayer(
                        in_features=attn_module.to_v.in_features,
                        out_features=attn_module.to_v.out_features,
                        rank=args.lora_r,
                        network_alpha=args.lora_alpha,
                        use_mask=is_cross,
                    )
                )
                attn_module.to_out[0].set_lora_layer(
                    TARALinearLayer(
                        in_features=attn_module.to_out[0].in_features,
                        out_features=attn_module.to_out[0].out_features,
                        rank=args.lora_r,
                        network_alpha=args.lora_alpha,
                        use_mask=False,
                    )
                )
                unet_lora_parameters.extend(attn_module.to_q.lora_layer.parameters())
                unet_lora_parameters.extend(attn_module.to_k.lora_layer.parameters())
                unet_lora_parameters.extend(attn_module.to_v.lora_layer.parameters())
                unet_lora_parameters.extend(attn_module.to_out[0].lora_layer.parameters())

                                                                           
            for name, param in unet.named_parameters():
                if "lora_layer" not in name:
                    param.requires_grad = False
                                                                          
            logger.info(f"UNet LoRA trainable params: {sum(p.numel() for p in unet_lora_parameters)}")

        else:
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=UNET_TARGET_MODULES,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
            )
            unet = get_peft_model(unet, config)
            unet.print_trainable_parameters()
            print(unet)

    ENABLE_LORA_CAPTURE = True
    ENABLE_BASE_CAPTURE = True

    lora_capture = {"k": [], "v": []}
    base_capture = {"k": []}

                            
    def _make_lora_hook(kind: str):
        def _hook(module, inp, out):
            if not ENABLE_LORA_CAPTURE:
                return
            x = inp[0]
            if x.dim() == 3:
                b, t, d = x.shape
                base = F.linear(x.view(-1, d), module.weight, module.bias).view(b, t, -1)
            else:
                base = F.linear(x, module.weight, module.bias)

            delta_masked = out - base   
            lora_capture[kind].append(delta_masked)
        return _hook

    def _make_base_hook(kind: str):
        def _hook(module, inp, _out):
            if not ENABLE_BASE_CAPTURE:
                return
            x = inp[0]
            if x.dim() == 3:
                b, t, d = x.shape
                base = F.linear(x.view(-1, d), module.weight, module.bias).view(b, t, -1)
            else:
                base = F.linear(x, module.weight, module.bias)
            base_capture[kind].append(base)
        return _hook

                                                         
    for _attn_name, _ in unet.attn_processors.items():
        if ".attn2" not in _attn_name:
            continue
        _attn = unet
        for _seg in _attn_name.split(".")[:-1]:
            _attn = getattr(_attn, _seg)
        for _layer, kind in [(_attn.to_k, "k"), (_attn.to_v, "v")]:
            _layer.register_forward_hook(_make_lora_hook(kind))
        _attn.to_k.register_forward_hook(_make_base_hook("k"))

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
    elif args.train_text_encoder and args.use_lora:
        config = LoraConfig(
            r=args.lora_text_encoder_r,
            lora_alpha=args.lora_text_encoder_alpha,
            target_modules=TEXT_ENCODER_TARGET_MODULES,
            lora_dropout=args.lora_text_encoder_dropout,
            bias=args.lora_text_encoder_bias,
        )
        text_encoder = get_peft_model(text_encoder, config)
        text_encoder.print_trainable_parameters()
        print(text_encoder)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
                                                          
        if args.train_text_encoder and not args.use_lora:
            text_encoder.gradient_checkpointing_enable()

                                                     
                                                                                              
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

                                                                                  
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

                        
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

                                       
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.num_dataloader_workers,
    )

                                                             
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

                                                
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

                                                                                             
                                                                                                     
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

                                                                  
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

                                                                                                              
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
                                                             
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

                                                                                  
                                                                 
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

            
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

                                                                     
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
                                           
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = resume_global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % num_update_steps_per_epoch

                                                      
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")


    def build_token_masks(tokenizer, input_ids, rare_word, super_word, device):
        """
        prompts:  list[str]  or  list of length-B Python strings
        return:   rare_mask, super_mask  —  shape (B, T) bool
        """
        input_ids = input_ids.to(device)

        rare_ids  = tokenizer.encode(" " + rare_word, add_special_tokens=False)
        super_ids = tokenizer.encode(" " + super_word, add_special_tokens=False)
        rare_ids  = torch.tensor(rare_ids,  device=device)
        super_ids = torch.tensor(super_ids, device=device)

        B, T = input_ids.shape
        rare_mask  = torch.zeros((B, T), dtype=torch.bool, device=device)
        super_mask = torch.zeros_like(rare_mask)

        for b in range(B):
            for j in range(T - len(rare_ids) + 1):
                if (input_ids[b, j : j+len(rare_ids)] == rare_ids).all():
                    rare_mask[b, j : j+len(rare_ids)] = True
            for j in range(T - len(super_ids) + 1):
                if (input_ids[b, j : j+len(super_ids)] == super_ids).all():
                    super_mask[b, j : j+len(super_ids)] = True

        return rare_mask, super_mask  

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        with TorchTracemalloc() if not args.no_tracemalloc else nullcontext() as tracemalloc:
            for step, batch in enumerate(train_dataloader):
                                                            
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        if args.report_to == "wandb":
                            accelerator.print(progress_bar)
                    continue

                with accelerator.accumulate(unet):
                    device = "cuda" if accelerator.device.type == "cuda" else "cpu"
                    input_ids = batch["input_ids"]
                    rare_mask, super_mask = build_token_masks(
                        tokenizer, 
                        input_ids,
                        rare_word=args.rare_word,
                        super_word=args.class_word,
                        device=device
                    )
                    if args.focus:
                        TARALinearLayer.set_mask(rare_mask.unsqueeze(-1))
                    else:
                        TARALinearLayer.set_mask(None)

                                                    
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215

                
                                                                
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                                                             
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    )
                    timesteps = timesteps.long()

                                                                                                
                                                             
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                                                             
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                                                
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                                                                              
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    if args.with_prior_preservation:
                                                                                                                     
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                                            
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                                                                  
                        loss = loss + args.prior_loss_weight * prior_loss

                    focus_loss = None
                    if args.focus and (lora_capture["k"] or lora_capture["v"]):
                        focus_loss_sum, n_terms = 0.0, 0
                        non_rare_mask = ~rare_mask
                        mask = non_rare_mask.unsqueeze(-1)
                        for delta in itertools.chain(lora_capture["k"], lora_capture["v"]):
                                                                                
                            delta = delta * mask
                            delta_loss = delta.abs().mean()
                            focus_loss_sum += delta_loss
                            n_terms += 1
                            del delta_loss, delta
                        focus_loss = focus_loss_sum / n_terms
                    
                    align_loss = None
                    if args.with_align_loss and base_capture["k"]:
                        align_sum, n_terms = 0.0, 0
                        mask_rare = rare_mask.unsqueeze(-1)                   
                        mask_class = super_mask.unsqueeze(-1)                    

                                                                           
                        for base_out, delta_out in zip(base_capture["k"], lora_capture["k"]):
                            total_rare = (base_out + delta_out) * mask_rare
                            base_cls = base_out * mask_class

                            if mask_rare.any() and mask_class.any():
                                rare_vec = total_rare[mask_rare.squeeze(-1)].view(-1, total_rare.size(-1))
                                cls_vec  = base_cls[mask_class.squeeze(-1)].view(-1, base_cls.size(-1))
                                align_sum += (rare_vec - cls_vec).abs().mean()
                                n_terms += 1

                        if n_terms > 0:
                            align_loss = align_sum / n_terms
                            loss = loss + args.align_loss_weight * align_loss
                    
                    for _lst in base_capture.values():
                                _lst.clear()
                    for _lst in lora_capture.values():
                                _lst.clear()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if args.train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                                                                                                
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    if args.report_to == "wandb":
                        accelerator.print(progress_bar)
                    global_step += 1

                                                                     
                                                         
                                                                                                    
                                                               
                                                                        

                logs = {"loss": loss.detach().item(), 
                "focus_loss": focus_loss.detach().item() if focus_loss is not None else -1, 
                "align_loss": align_loss.detach().item() if align_loss is not None else -1,
                "lr": lr_scheduler.get_last_lr()[0]}

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if (
                    args.validation_prompt is not None
                    and (step + num_update_steps_per_epoch * epoch) % args.validation_steps == 0
                ):
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {args.validation_prompt}."
                    )
                                     
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        safety_checker=None,
                        revision=args.revision,
                    )
                                                                                      
                                                                       
                    pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
                    pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                                   
                    if args.seed is not None:
                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    else:
                        generator = None
                    images = []
                    for _ in range(args.num_validation_images):
                        image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                        images.append(image)

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            import wandb

                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )

                    del pipeline
                    torch.cuda.empty_cache()

                if global_step >= args.max_train_steps:
                    break
                                                                                                             

                                                                      
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_lora:
            unwarpped_unet = accelerator.unwrap_model(unet)
            unwarpped_unet.save_pretrained(
                os.path.join(args.output_dir, "unet"), state_dict=accelerator.get_state_dict(unet)
            )
            if args.train_text_encoder:
                unwarpped_text_encoder = accelerator.unwrap_model(text_encoder)
                unwarpped_text_encoder.save_pretrained(
                    os.path.join(args.output_dir, "text_encoder"),
                    state_dict=accelerator.get_state_dict(text_encoder),
                )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                revision=args.revision,
            )
            pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            api.upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                run_as_future=True,
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)