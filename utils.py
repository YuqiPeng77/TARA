import torch.nn as nn
import os
import re
from typing import Optional, Dict, ClassVar
from huggingface_hub import hf_hub_download
import torch
from safetensors import safe_open
from diffusers.models.lora import LoRACompatibleLinear
import torch
from typing import List, Tuple


LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"


def get_lora_weights(
    lora_name_or_path: str,
    subfolder: Optional[str] = None,
    sub_lora_weights_name: str = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Args:
        lora_name_or_path (str): huggingface repo id or folder path of lora weights
        subfolder (Optional[str], optional): sub folder. Defaults to None.
    """
    if os.path.exists(lora_name_or_path):
        if subfolder is not None:
            lora_name_or_path = os.path.join(lora_name_or_path, subfolder)
        if os.path.isdir(lora_name_or_path):
            lora_name_or_path = os.path.join(lora_name_or_path, LORA_WEIGHT_NAME_SAFE)
    else:
        lora_name_or_path = hf_hub_download(
            repo_id=lora_name_or_path,
            filename=(
                sub_lora_weights_name
                if sub_lora_weights_name is not None
                else LORA_WEIGHT_NAME_SAFE
            ),
            subfolder=subfolder,
            **kwargs,
        )
    assert lora_name_or_path.endswith(
        ".safetensors"
    ), "Currently only safetensors is supported"
    tensors = {}
    with safe_open(lora_name_or_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def merge_lora_weights(
                                                                 
    tensors: torch.Tensor, key: str, prefix: str = "unet."
) -> Dict[str, torch.Tensor]:
    """
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict. Defaults to "unet.unet.".
    """
    target_key = prefix + key
    out1 = {}
    out2 = {}
    for part in ["to_q", "to_k", "to_v", "to_out.0"]:
        down_key = target_key + f".{part}.lora.down.weight"
        up_key = target_key + f".{part}.lora.up.weight"
        out1[part] = tensors[up_key]
        out2[part] = tensors[down_key]
    return out1, out2


def merge_sd_lora_weights(
    tensors: torch.Tensor, key: str, prefix: str = "unet.unet."
) -> Dict[str, torch.Tensor]:
    """
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict.
    """

    target_key = prefix + key
    out1 = {}
    out2 = {}

    for part in ["to_q", "to_k", "to_v", "to_out.0"]:
        down_key = target_key + f".{part}.lora.down.weight"
        up_key = target_key + f".{part}.lora.up.weight"
        out1[part] = tensors[up_key]
        out2[part] = tensors[down_key]
    return out1, out2


                                                                            
def insert_sd_two_lora_to_unet(
    unet,
    lora_weights_path_1: str,
    lora_weights_path_2: str,
    weight1: float = 1,
    weight2: float = 1,
    prefix: str = "unet.",
):
    """
    Inject two LoRA checkpoints into UNet by *adding* their merged weight deltas.

    Args:
        unet:            The target UNet model.
        lora_weights_path_1 (str): Path to the first LoRA (.safetensors).
        lora_weights_path_2 (str): Path to the second LoRA (.safetensors).
        weight1 (float): Scaling factor for the first LoRA delta. Default 1.
        weight2 (float): Scaling factor for the second LoRA delta. Default 1.
        prefix (str):    Prefix of the LoRA keys inside the safetensors file.

    Returns:
        The UNet model with LoRA deltas applied as `LoRALinearLayerInference`.
    """
                                
    lora_weights_1 = get_lora_weights(lora_weights_path_1)
    lora_weights_2 = get_lora_weights(lora_weights_path_2)

                                                              
    for attn_processor_name, _ in unet.attn_processors.items():
                                                                             
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        attn_name = ".".join(attn_processor_name.split(".")[:-1])

                                                                              
        for sub_name in ["to_q", "to_k", "to_v"]:
            old_layer = getattr(attn_module, sub_name)
            if not isinstance(old_layer, LoRACompatibleLinear):
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
                setattr(attn_module, sub_name, new_layer)

                                                                                       
        old_layer = attn_module.to_out[0]
        if not isinstance(old_layer, LoRACompatibleLinear):
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

                                                                              
        for part, layer_accessor in [
            ("to_q", lambda m: m.to_q),
            ("to_k", lambda m: m.to_k),
            ("to_v", lambda m: m.to_v),
            ("to_out.0", lambda m: m.to_out[0]),
        ]:
            up1, down1 = extract_lora_weights(lora_weights_1, attn_name=attn_name, part=part)
            up2, down2 = extract_lora_weights(lora_weights_2, attn_name=attn_name, part=part)

                                     
            delta1 = up1 @ down1
            delta2 = up2 @ down2

                                                      
            merged_weight = weight1 * delta1 + weight2 * delta2

            layer = layer_accessor(attn_module)
            layer.set_lora_layer(LoRALinearLayerInference(weight=merged_weight))

    return unet



def insert_sd_lora_list_to_unet(
    unet,
    lora_list: list,
    weight_list: list = [],
    prefix: str = "unet.",
):
                                         
    merged_weights = {}

    for lora in lora_list:
        weights = get_lora_weights(lora)
        for name, delta in weights.items():
            if name not in merged_weights:
                merged_weights[name] = delta.to(unet.device)
            else:
                merged_weights[name] += delta.to(unet.device)

                                           
    device = next(unet.parameters()).device

                                                              
    for attn_processor_name, _ in unet.attn_processors.items():
                                                                             
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        attn_name = ".".join(attn_processor_name.split(".")[:-1])

                                                                              
        for sub_name in ["to_q", "to_k", "to_v"]:
            old_layer = getattr(attn_module, sub_name)
            if not isinstance(old_layer, LoRACompatibleLinear):
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
                setattr(attn_module, sub_name, new_layer)

                                                                                       
        old_layer = attn_module.to_out[0]
        if not isinstance(old_layer, LoRACompatibleLinear):
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

                                                                              
        for part, layer_accessor in [
            ("to_q", lambda m: m.to_q),
            ("to_k", lambda m: m.to_k),
            ("to_v", lambda m: m.to_v),
            ("to_out.0", lambda m: m.to_out[0]),
        ]:
                                                                                      
            try:
                up, down = extract_lora_weights(merged_weights, attn_name=attn_name, part=part)
            except KeyError:
                continue
            up = up.to(device)
            down = down.to(device)
            merged_weight = up @ down
            layer = layer_accessor(attn_module)
            merged_weight = merged_weight.to(device)
            layer.set_lora_layer(LoRALinearLayerInference(weight=merged_weight))

    return unet


    
                                    
def extract_lora_weights(lora_weights, attn_name, part, prefix="unet."):
    key_a = f"{prefix}{attn_name}.{part}.lora.down.weight"
    key_b = f"{prefix}{attn_name}.{part}.lora.up.weight"
    if key_a in lora_weights or key_b in lora_weights:
        return lora_weights[key_b], lora_weights[key_a]
    key_a = key_a.replace("unet.", "base_model.model.").replace("lora.down.weight", "lora_A.weight")
    key_b = key_b.replace("unet.", "base_model.model.").replace("lora.up.weight", "lora_B.weight")
    if key_a in lora_weights or key_b in lora_weights:
        return lora_weights[key_b], lora_weights[key_a]
    key_a = f"{attn_name}.{part}.lora_layer.down.weight"
    key_b = f"{attn_name}.{part}.lora_layer.up.weight"
    if key_a in lora_weights or key_b in lora_weights:
        return lora_weights[key_b], lora_weights[key_a]
    
    raise KeyError(f"Missing LoRA weights for {attn_name}.{part}")

                                                                                       
def insert_sd_lora_to_unet(unet, lora_weights_path, prefix="unet."):
    lora_weights = get_lora_weights(lora_weights_path)

    for attn_processor_name, attn_processor in unet.attn_processors.items():
                                     
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        attn_name = ".".join(attn_processor_name.split(".")[:-1])


                                                                                          
                             
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


        lora_up, lora_down = extract_lora_weights(lora_weights, attn_name=attn_name, part="to_q")
        merged_weight = lora_up @ lora_down
        layer = attn_module.to_q
        layer.set_lora_layer(
            LoRALinearLayerInference(weight=merged_weight)
        )

        lora_up, lora_down = extract_lora_weights(lora_weights, attn_name=attn_name, part="to_k")
        merged_weight = lora_up @ lora_down
        layer = attn_module.to_k
        layer.set_lora_layer(
            LoRALinearLayerInference(weight=merged_weight)
        )

        lora_up, lora_down = extract_lora_weights(lora_weights, attn_name=attn_name, part="to_v")
        merged_weight = lora_up @ lora_down
        layer = attn_module.to_v
        layer.set_lora_layer(
            LoRALinearLayerInference(weight=merged_weight)
        )

        lora_up, lora_down = extract_lora_weights(lora_weights, attn_name=attn_name, part="to_out.0")
        merged_weight = lora_up @ lora_down
        layer = attn_module.to_out[0]
        layer.set_lora_layer(
            LoRALinearLayerInference(weight=merged_weight)
        )

    return unet



def build_token_masks(
    tokenizer,
    prompts: List[str],                             
    rare_word: str,                           
    device: torch.device,                 
) -> torch.Tensor:
                                                                        
    rare_word  = rare_word  or ""
                     
    enc = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc.input_ids.to(device)                 

                              
    rare_ids  = tokenizer.encode(" " + rare_word,  add_special_tokens=False)
    rare_ids  = torch.tensor(rare_ids,  device=device)

    B, T = input_ids.shape
    rare_mask  = torch.zeros((B, T), dtype=torch.bool, device=device)

    if rare_ids.numel() == 0:
        return rare_mask

    L = rare_ids.shape[0]
    for b in range(B):
        for j in range(T - L + 1):
            if (input_ids[b, j : j + L] == rare_ids).all():
                rare_mask[b, j : j + L] = True

    return rare_mask


                                       
                                                                                          
    
                                                                        
                                                                    

                                                          
                                                  
       
                                                                                
                                                                            

                                                                              
                                       
                            
                                                       
                                                   
                                                                   
                                                                        
                                                                    
           
                                                                        
                                                                  
           
                    
                             
                           
                                             
                                             
                                                             
                                                             
                                                             
                                                             
           
                                                                             
                                          
                                     
                           
                              
                                                           
                                                             
               
           
                                          
                                     
                           
                              
                                                           
                                                             
               
           
                                          
                                     
                           
                              
                                                           
                                                             
               
           
                                               
                                     
                           
                                  
                                                                
                                                                  
               
           
                 


                                    
             
                  
                       
                           
                         
            
           
                   
                    
             
    
                          
                            
                                                   
                                                       
                              
           
                                                 
       
                        
                                                                      
                                      
       
                        
                                                                      
                                      
       
                              
                                                                    
                                      
       
                              
                                                                    
                                      
       

                                              
                                                             
                                            
                                                      
                                                        
                        
                                                   
                                                 
           

                                         
                          
                        
                                         
                                                                   
                                                                    
                                                         
                                                         
                                                               
                                                               
                                          
                             
           
                                                

                                                                   
                                                               
                                                           

                        


                                         
           
                                
                              
            
           
                   
                                           
                                         
                         
    
                                              
                                                      
                                                         
       
                                            
                                                    
                                                       
       
                                                          
                                                  
       
                                                         
                                                     
                                                       

                                                
                                                                                         

                                   
                                 
                                                                                  
                                
                                                           
                                                       
                                                                       
                                                                            
                                                    
                                                   
                                    
                                                   
                   
               
                                                                            
                                                    
                                                 
                                    
                                                 
                   
               
                        
                                 
                               
                                                 
                                                 
                                   
                                                                 
                                                                 
                                                                 
                                                                 
               
                                                                                 
                                                                
                                                                
                                                                

                                          
                                                           
                                                             
                            
                                                        
                                                      
               
                                                                     
                                                                 
                                     

                                              
                                         
                               
                                  
                                                               
                                                                 
                   
               
                                              
                                         
                               
                                  
                                                               
                                                                 
                   
               
                                              
                                         
                               
                                  
                                                               
                                                                 
                   
               

                                             
                                                       
                                             
                                   
                                          
                                                                        
                                                                          
                       
                   

                                     
                               
                                 
                                               
                                                                             
                    
                                                                                  
                                            
                                
                                         
                                                            
                                                            
                                                        
                              
                            
                                             
                                              
                                
               
                    
                                          
                                            
                                          
                                         
                                                            
                                                            
                                                        
                              
                            
                                             
                                              
                                
               
                    
                                          
                                            
                                          
                                         
                                              
                                                            
                                                        
                              
                            
                                             
                                              
                                
               

                                        
                                                                      
                                                                           
                    
                                          
                                            
                                          
                                         
                                      
                                   
                                   
                                   
                                   
                             
                             
                             
                    
                                                            
                                                        
                              
                            
                                             
                                              
                                
               
                                                    
                                            
                                                  
                                         
                                        
                                                            
                                                        
                              
                            
                                             
                                              
                                
               
                     
                                           
                                            
                                           
                                         
                                              
                                                            
                                                        
                              
                            
                                             
                                              
                                
               
                             
                                           
                                            
                                           
                                         
                                              
                                                            
                                                        
                              
                            
                                             
                                              
                                
               
                  
                                                                        
                                
                                                    
                                                                       
                                                     
                                                    
                                                                    
                                                                
                                      
                                    
                                                     
                                                      
                                        
                       
                                
                                                    
                                                                 
                                                     
                                                
                                                                    
                                                                
                                        
                                    
                                                     
                                                      
                                        
                       
                          
                                                                                
                                
                                                    
                                                                               
                                                     
                                                    
                                                                    
                                                                
                                      
                                    
                                                     
                                                      
                                        
                       
                                
                                                    
                                                                         
                                                     
                                                
                                                                    
                                                                
                                      
                                    
                                                     
                                                       
                                        
                       

           
                                                                                                                      
                 


def rename_safetensors_layer_name(tensor):
    def rename_key(key):
        patten = r"(\w+\_blocks\.\d+\.attentions\.\d+\.+\w+\.\d+\.attn\d+)"
        match = re.search(patten, key)
        new_key = "unet.unet."
        if match:
            new_key += match.group(1)
            new_key += "."

        patten = r"(\w+\_block\.attentions\.\d+\.+\w+\.\d+\.attn\d+)"
        match = re.search(patten, key)
        if match:
            new_key += match.group(1)
            new_key += "."

        key = key.replace(".", "_")
        if "to_q" in key:
            new_key += "to_q"
        elif "to_k" in key:
            new_key += "to_k"
        elif "to_v" in key:
            new_key += "to_v"
        elif "to_out" in key:
            new_key += "to_out.0"
        if "down_weight" in key:
            new_key += ".lora.down.weight"
        if "up_weight" in key:
            new_key += ".lora.up.weight"
        return new_key

    renamed_tensor = {rename_key(key): value for key, value in tensor.items()}

    return renamed_tensor




                                     
from typing import Optional, Union

class LoRALinearLayerInference(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(weight.to(device=device, dtype=dtype), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        hidden_states = nn.functional.linear(
            hidden_states.to(self.weight.dtype), weight=self.weight
        )
        return hidden_states.to(orig_dtype)
    
                                                                               
                                                           
                                                                               
def insert_sd_tara_to_unet(unet, lora_weights_path, mask: Optional[torch.Tensor] = None, prefix: str = "unet."):
    """
    与 insert_sd_lora_to_unet 类似，但使用 TARALinearLayerInference。
    仅当层名含 `.attn2` 时，对 to_k / to_v 启用掩码。
    使用示例:
        TARALinearLayerInference.set_mask(rare_mask.unsqueeze(-1))
        images = pipe(...)
        TARALinearLayerInference.set_mask(None)
    """
    lora_weights = get_lora_weights(lora_weights_path)

    for attn_processor_name, _ in unet.attn_processors.items():
                                
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        attn_name = ".".join(attn_processor_name.split(".")[:-1])

        is_cross = ".attn2" in attn_processor_name            

                                                                   
        for sub in ["to_q", "to_k", "to_v"]:
            layer = getattr(attn_module, sub)
            if not isinstance(layer, LoRACompatibleLinear):
                rep = LoRACompatibleLinear(
                    in_features=layer.in_features,
                    out_features=layer.out_features,
                    bias=layer.bias is not None,
                    device=layer.weight.device,
                    dtype=layer.weight.dtype,
                )
                rep.weight.data.copy_(layer.weight.data)
                if layer.bias is not None:
                    rep.bias.data.copy_(layer.bias.data)
                setattr(attn_module, sub, rep)

                   
        layer = attn_module.to_out[0]
        if not isinstance(layer, LoRACompatibleLinear):
            rep = LoRACompatibleLinear(
                in_features=layer.in_features,
                out_features=layer.out_features,
                bias=layer.bias is not None,
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            rep.weight.data.copy_(layer.weight.data)
            if layer.bias is not None:
                rep.bias.data.copy_(layer.bias.data)
            attn_module.to_out[0] = rep

                                                                       
        for part, layer in [
            ("to_q", attn_module.to_q),
            ("to_k", attn_module.to_k),
            ("to_v", attn_module.to_v),
            ("to_out.0", attn_module.to_out[0]),
        ]:
            try:
                up, down = extract_lora_weights(
                    lora_weights, attn_name=attn_name, part=part, prefix=prefix
                )
            except KeyError:
                continue                       
            merged = up @ down
            use_mask = is_cross and part in ["to_k", "to_v"]
            layer.set_lora_layer(
                TARALinearLayerInference(weight=merged, use_mask=use_mask, mask=mask if use_mask else None,            
                )
            )
    return unet
    
                                                                       
                                                                    
                                                                       
class TARALinearLayerInference(nn.Module):
    """
    每个层实例可单独带一个 `mask` (形状可广播到 delta)：
        1 = 保留 rare-token LoRA 增量
        0 = 清零普通 token LoRA 增量
    若 `mask=None` 或 `use_mask=False` → 行为与普通 LoRA 一致。
    """

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        use_mask: bool = False,
        mask: Optional[torch.Tensor] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            weight.to(device=device, dtype=dtype), requires_grad=False
        )
        self.use_mask = use_mask
        self.mask = mask                  

                                                                   
    def set_mask(self, mask: Optional[torch.Tensor]) -> None:
        """推理过程中可随时更换 mask。"""
        self.mask = mask

                                                                       
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:                          
        orig_dtype = hidden_states.dtype
        delta = nn.functional.linear(
            hidden_states.to(self.weight.dtype), weight=self.weight
        )
        if self.use_mask and self.mask is not None:
            delta = delta * self.mask.to(delta.dtype)
        return delta.to(orig_dtype)
    


class MultiTARALinearLayerInference(nn.Module):

    def __init__(self, layers=None):
        super().__init__()
        self.layers = nn.ModuleList(layers or [])

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        delta = 0
        for layer in self.layers:
            delta = delta + layer(x)
        return delta


def _append_tara_layer(base_linear, new_tara):
    if hasattr(base_linear, "lora_layer") and base_linear.lora_layer is not None:
        if isinstance(base_linear.lora_layer, MultiTARALinearLayerInference):
            base_linear.lora_layer.add_layer(new_tara)
        else:
            multi = MultiTARALinearLayerInference([base_linear.lora_layer, new_tara])
            base_linear.lora_layer = multi
    else:
        base_linear.set_lora_layer(new_tara)


                                                                               
                                                                 
                                                                               
def insert_multi_sd_tara_to_unet(
    unet,
    lora_paths: List[str],
    mask_list: List[Optional[torch.Tensor]],
    prefix: str = "unet.",
):
    if len(lora_paths) != len(mask_list):
        raise ValueError("`lora_paths` and `mask_list` must have the same length.")

                                                                              
    for attn_processor_name, _ in unet.attn_processors.items():
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        for sub in ["to_q", "to_k", "to_v"]:
            old_layer = getattr(attn_module, sub)
            if not isinstance(old_layer, LoRACompatibleLinear):
                rep = LoRACompatibleLinear(
                    in_features=old_layer.in_features,
                    out_features=old_layer.out_features,
                    bias=old_layer.bias is not None,
                    device=old_layer.weight.device,
                    dtype=old_layer.weight.dtype,
                )
                rep.weight.data.copy_(old_layer.weight.data)
                if old_layer.bias is not None:
                    rep.bias.data.copy_(old_layer.bias.data)
                setattr(attn_module, sub, rep)

        old_layer = attn_module.to_out[0]
        if not isinstance(old_layer, LoRACompatibleLinear):
            rep = LoRACompatibleLinear(
                in_features=old_layer.in_features,
                out_features=old_layer.out_features,
                bias=old_layer.bias is not None,
                device=old_layer.weight.device,
                dtype=old_layer.weight.dtype,
            )
            rep.weight.data.copy_(old_layer.weight.data)
            if old_layer.bias is not None:
                rep.bias.data.copy_(old_layer.bias.data)
            attn_module.to_out[0] = rep

                                                                       
                                                                           
                                                                       
    for tara_path, tara_mask in zip(lora_paths, mask_list):
        lora_weights = get_lora_weights(tara_path)

        for attn_processor_name, _ in unet.attn_processors.items():
            attn_module = unet
            for n in attn_processor_name.split(".")[:-1]:
                attn_module = getattr(attn_module, n)
            attn_name = ".".join(attn_processor_name.split(".")[:-1])

            is_cross = ".attn2" in attn_processor_name

            for part, layer in [
                ("to_q", attn_module.to_q),
                ("to_k", attn_module.to_k),
                ("to_v", attn_module.to_v),
                ("to_out.0", attn_module.to_out[0]),
            ]:
                try:
                    up, down = extract_lora_weights(
                        lora_weights, attn_name=attn_name, part=part, prefix=prefix
                    )
                except KeyError:
                    continue                                               
                merged = up @ down
                use_mask = is_cross and part in ["to_k", "to_v"]
                new_tara = TARALinearLayerInference(
                    weight=merged,
                    use_mask=use_mask,
                    mask=tara_mask if use_mask else None,
                )
                _append_tara_layer(layer, new_tara)

    return unet



    

from pathlib import Path
from typing import List, Type, Any, Dict, Tuple, Union
import math

from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F


class UNetCrossAttentionHooker():
    def __init__(
            self,
            is_train:bool=True,
    ):
        self.cross_attn_maps=[]
        self.is_train=is_train
    
    def clear(self):
                                              
        for i in range(len(self.cross_attn_maps)):
            self.cross_attn_maps[i] = self.cross_attn_maps[i].detach()
        self.cross_attn_maps.clear()

    def _unravel_attn(self, x, n_heads):
                                                   
                                                  
                      
        """
        Unravels the attention, returning it as a collection of heat maps.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.

        Returns:
            `List[Tuple[int, torch.Tensor]]`: the list of heat maps across heads.
        """
        maps = []
        x = x.permute(2, 0, 1)

        for map_ in x:
            if not self.is_train:
                map_ = map_[map_.size(0) // 2:]                            
            maps.append(map_)

        maps = torch.stack(maps, 0)                                        
        maps=maps.permute(1,0,2)                                       
        maps=maps.reshape([maps.shape[0]//n_heads,n_heads,*maps.shape[1:]])                                                   
        maps=maps.permute(0,2,1,3)                                                   
        return maps

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
    ):
        """Capture attentions and aggregate them."""
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attn=(encoder_hidden_states is not None)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if is_cross_attn:
                                                                 
            maps = self._unravel_attn(attention_probs, attn.heads)

                                                                          
            maps = maps.to(torch.float16)
                                                    
            self.cross_attn_maps.append(maps)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

                     
        hidden_states = attn.to_out[0](hidden_states)
                 
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states



