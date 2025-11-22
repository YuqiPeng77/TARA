from diffusers.models.lora import LoRALinearLayer
import torch
import torch.nn as nn
from typing import Optional, Union
import math
import torch.nn.functional as F
from typing import Optional, ClassVar

                                                                     
                                                                           
class TARALinearLayer(LoRALinearLayer):

    _active_mask: ClassVar[Optional[torch.Tensor]] = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        *,
        use_mask: bool = False,
    ):
        super().__init__(
            in_features,
            out_features,
            rank=rank,
            network_alpha=network_alpha,
            device=device,
            dtype=dtype,
        )
        self.use_mask = use_mask

                                                                                   
    @classmethod
    def set_mask(cls, mask: Optional[torch.Tensor]) -> None:
        cls._active_mask = mask

                                                        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:                          
        delta = super().forward(hidden_states)
        if self.use_mask and TARALinearLayer._active_mask is not None:
            delta = delta * TARALinearLayer._active_mask.to(delta.dtype)
        return delta