from typing import *

import src.constitutive_models.physical_constitutive_models as physical
from .neural_constitutive_models import *

def get_elasticity(elasticity_name: str, physicals: Optional[List[str]]=None, device: str='cuda'):
    if elasticity_name == 'neural':
        return GumbelElasticity(
            physicals=physicals,
            device=device
        )
    return PresetElasticity(elasticity_name, device=device)

def get_plasticity(plasticity_name: str, physicals: Optional[List[str]]=None, device: str='cuda'):
    if plasticity_name == 'neural':
        return GumbelPlasticity(
            physicals=physicals,
            device=device
        )
    return PresetPlasticity(plasticity_name, device=device)

