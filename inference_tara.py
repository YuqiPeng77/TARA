import argparse
from diffusers import DiffusionPipeline
import torch
import os
from utils import (
    build_token_masks,
    insert_multi_sd_tara_to_unet,
)
import json
import sys

                                                                                     
                                                                       

                                                                   
def build_inference_mask(args, pipe, rare_token=None, device="cpu"):
    prompts = [args.prompt]  
    def _mask_for_tokenizer(tokenizer, rare_token):
                                                             
        if rare_token is not None:
            mask = build_token_masks(
                tokenizer, 
                prompts,
                rare_word=rare_token,
                device=device,
            )
        return mask

    if args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-v1-5":
        mask = _mask_for_tokenizer(pipe.tokenizer, rare_token)
        mask = mask.unsqueeze(-1)  
        return mask
    else:
        mask1 = _mask_for_tokenizer(pipe.tokenizer, rare_token)
        mask2 = _mask_for_tokenizer(pipe.tokenizer_2, rare_token)
        mask = mask1 | mask2
        mask = mask.unsqueeze(-1)
        return mask


                              
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-v1-5",
        help="Pretrained model path",
    )
    parser.add_argument(
        "--lora_list",
        type=json.loads,
        help="JSON-encoded list of LoRA paths (e.g., \"[\"/path/1\", \"/path/2\"]\")",
        default=None,
    )
    parser.add_argument(
        "--rare_token_list",
        type=json.loads,
        help="JSON-encoded list of rare tokens matching lora_list",
        default=None,
    )
    parser.add_argument(
        "--number",
        type=int,
        help="Number of images to generate",
        default=6,
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Output folder path",
        default="output",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for the image generation",
        default="a sbu cat in szn style",
    )
    return parser.parse_args()


args = parse_args()

                                               
pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)


device = "cuda" if torch.cuda.is_available() else "cpu"
                                             
if args.lora_list and args.rare_token_list:
    mask_list = []
    for word in args.rare_token_list:
        mask_list.append(build_inference_mask(args=args, pipe=pipe, rare_token=word, device=device))
    pipe.unet = insert_multi_sd_tara_to_unet(
            pipe.unet, args.lora_list, mask_list
        )


                                                        
pipe.to(device, dtype=torch.float16)

                                   
def run():
    seeds = list(range(args.number))
    seeds = [see for see in seeds]
    
    os.makedirs(args.output_folder, exist_ok=True)

    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(prompt=args.prompt, generator=generator).images[0]
        output_path = os.path.join(args.output_folder, f"output_image_{index}.png")
        image.save(output_path)
        print(output_path)


                            
if __name__ == "__main__":
    run()
