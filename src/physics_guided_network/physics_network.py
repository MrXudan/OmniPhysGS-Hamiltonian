from typing import *

import torch
from torch import Tensor
from torch import nn
from torch.nn import Module
from omegaconf import DictConfig

from .knn_transformer import KNNTransformer

class PhysicsNetwork(Module):
    def __init__(
        self,
        elasticity_physicals: List[str],
        plasticity_physicals: List[str],
        in_channels: int,
        params: Optional[DictConfig] = None,
        n_particles: Optional[int] = None,
        export_path: Optional[str] = None
    ):
        super(PhysicsNetwork, self).__init__()
        
        self.elasticity_physicals = elasticity_physicals
        self.plasticity_physicals = plasticity_physicals
        self.elasticity_dim = len(elasticity_physicals)
        self.plasticity_dim = len(plasticity_physicals)
        
        network = params.network
        if network == 'knn':
            self.physics_network = KNNTransformer(
                elasticity_dim=self.elasticity_dim,
                plasticity_dim=self.plasticity_dim,
                in_channels=in_channels,
                num_groups=params.num_groups,
                group_size=params.group_size,
                hidden_size=params.hidden_size, 
                depth=params.depth,
                num_heads=params.num_heads,
                mlp_ratio=params.mlp_ratio,
                export_path=export_path
            )
        elif network == 'naive':
            assert n_particles is not None
            self.physics_network = NaiveNetwork(
                elasticity_dim=self.elasticity_dim,
                plasticity_dim=self.plasticity_dim,
                n_particles=n_particles
            )
        
        elif network == 'mlp':
            self.physics_network = MLPNetwork(
                elasticity_dim=self.elasticity_dim,
                plasticity_dim=self.plasticity_dim,
                in_channels=in_channels,
                hidden_size=params.hidden_size,
            )    
            
        else:
            raise NotImplementedError(f'Network {network} is not implemented.')
        
    def forward(
        self, 
        x: Tensor,
        features: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return self.physics_network(x, features)    

class NaiveNetwork(Module):
    def __init__(
        self,
        elasticity_dim: int,
        plasticity_dim: int,
        n_particles: int
    ):
        super(NaiveNetwork, self).__init__()
        
        self.e_cat = nn.Parameter( 1.0 / elasticity_dim * torch.ones(n_particles, elasticity_dim) )
        self.p_cat = nn.Parameter( 1.0 / plasticity_dim * torch.ones(n_particles, plasticity_dim) )
        
    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        return self.e_cat, self.p_cat

class MLPNetwork(Module):
    def __init__(
        self,
        elasticity_dim: int,
        plasticity_dim: int,
        in_channels: int,
        hidden_size: int=768,
    ):
        super(MLPNetwork, self).__init__()
        
        self.e_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, elasticity_dim)
        )
        
        self.p_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, plasticity_dim)
        )
        
    def forward(self, x: Tensor, features: Tensor) -> Tuple[Tensor, Tensor]:
        e_cat = self.e_mlp(features)
        p_cat = self.p_mlp(features)
        return e_cat, p_cat 
        
