from typing import *

from functools import partial
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .abstract import Elasticity, Plasticity
import src.constitutive_models.physical_constitutive_models as physical

class PresetElasticity(Elasticity):
    def __init__(
        self,
        elasticity_name: str,
        device: torch.device='cuda',
    ) -> None:
        super().__init__()
        self.elasticity_name = elasticity_name
        self.elasticity = getattr(physical, elasticity_name)().to(device)

    def forward(self, F: Tensor, *args, **kwargs) -> Tensor:
        return self.elasticity(F)
    
    def name(self) -> str:
        return f'[Elasticity] Physical {self.elasticity_name}'
    
class PresetPlasticity(Plasticity):
    def __init__(
        self,
        plasticity_name: str,
        device: torch.device='cuda',
    ) -> None:
        super().__init__()
        self.plasticity_name = plasticity_name
        self.plasticity = getattr(physical, plasticity_name)().to(device)

    def forward(self, F: Tensor, *args, **kwargs) -> Tensor:
        return self.plasticity(F)
    
    def name(self) -> str:
        return f'[Plasticity] Physical {self.plasticity_name}'
     
def hard_softmax(logits: Tensor, dim: int) -> Tensor:
    y_soft = logits.softmax(dim=dim)
    index = y_soft.argmax(dim=dim, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(dim=dim, index=index, value=1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret

class GumbelElasticity(Elasticity):
    def __init__(
        self,
        physicals: List[str],
        device: torch.device='cuda',
    ) -> None:
        super().__init__()
        self.physical_names = physicals
        self.physicals = nn.ModuleList([getattr(physical, p)().to(device) for p in physicals])
        self.category_dim = len(self.physicals)
        
    def forward(self, F: Tensor, elasticity_category: Tensor) -> Tensor:
        assert elasticity_category.shape[1] == self.category_dim
        # max_category = torch.functional.F.gumbel_softmax(elasticity_category, tau=1.0, dim=1, hard=True) # num_particles * category_dim
        max_category = hard_softmax(elasticity_category, dim=1)
        max_category = max_category.unsqueeze(dim=2).unsqueeze(dim=3) # num_particles * category_dim * 1 * 1
        possible_stress = [p(F) for p in self.physicals] # category_dim * num_particles * 3 * 3
        possible_stress = torch.stack(possible_stress, dim=1) # num_particles * category_dim * 3 * 3
        stress = possible_stress * max_category # num_particles * category_dim * 3 * 3
        stress = stress.sum(dim=1) # num_particles * 3 * 3
        return stress
    
    def name(self) -> str:
        return f'[Elasticity] Neural GumbelElasticity among {", ".join(self.physical_names)}'

class GumbelPlasticity(Plasticity):
    def __init__(
        self,
        physicals: List[str],
        device: torch.device='cuda',
    ) -> None:
        super().__init__()
        self.physical_names = physicals
        self.physicals = nn.ModuleList([getattr(physical, p)().to(device) for p in physicals])
        self.category_dim = len(self.physicals)
        
    def forward(self, F: Tensor, plasticity_category: Tensor) -> Tensor:
        assert plasticity_category.shape[1] == self.category_dim
        # max_category = torch.functional.F.gumbel_softmax(plasticity_category, tau=5.0, dim=1, hard=True) # num_particles * category_dim
        max_category = hard_softmax(plasticity_category, dim=1)
        max_category = max_category.unsqueeze(dim=2).unsqueeze(dim=3) # num_particles * category_dim * 1 * 1
        possible_stress = [p(F) for p in self.physicals] # category_dim * num_particles * 3 * 3
        possible_stress = torch.stack(possible_stress, dim=1) # num_particles * category_dim * 3 * 3
        stress = possible_stress * max_category # num_particles * category_dim * 3 * 3
        stress = stress.sum(dim=1) # num_particles * 3 * 3
        return stress   
      
    def name(self) -> str:
        return f'[Plasticity] Neural GumbelPlasticity among {", ".join(self.physical_names)}'