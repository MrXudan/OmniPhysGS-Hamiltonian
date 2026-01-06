from typing import *

import os
import torch
from torch import Tensor
from torch import nn
from torch.nn import Module

from src.utils.render_utils import particle_position_tensor_to_ply

class KNNTransformer(Module):
    
    def __init__(
        self,
        elasticity_dim: int,
        plasticity_dim: int,
        in_channels,
        num_groups: int = 2048,
        group_size: int = 32,
        hidden_size: int = 128,
        depth: int = 16,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        clamp_physical: bool = False,
        export_path: Optional[str] = None,
    ) -> None:
        super(KNNTransformer, self).__init__()
        
        self.grouper = Grouper(num_groups, group_size, export_path)
        self.group_encoder = GroupEncoder(in_channels, hidden_dim=hidden_size // 4, out_dim=hidden_size)
        
        # self.pos_emb = nn.Parameter(get_pos_emb(num_groups, hidden_size))
        self.pos_emb = nn.Parameter(torch.zeros(1, num_groups, hidden_size))
        
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        
        self.to_group_e_cat = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, elasticity_dim)
        )
        self.to_group_p_cat = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, plasticity_dim)
        )
        
        self.clamp_physical = clamp_physical
        self.cache = None # cache the group features
        
    def forward(
        self, 
        x: Tensor,
        features: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        
        # 1. Grouping the points based on the centers with FPS and KNN
        if self.cache is None:
            x = x.unsqueeze(0) # (N, 3) -> (1, N, 3)
            features = features.unsqueeze(0) # (N, F) -> (1, N, F)
            with torch.no_grad():
                neighbors, nearest_center_idx = self.grouper(x, features) # (1, N, 3), (1, N, F) -> (1, G, M, F), (1, N)
            neighbors = neighbors.detach() # avoid backpropagate to the FPS and KNN
            nearest_center_idx = nearest_center_idx.detach()
            self.cache = (neighbors, nearest_center_idx)
        else:
            neighbors, nearest_center_idx = self.cache
        # 2. Encode the groups
        encoded_features = self.group_encoder(neighbors) # (1, G, M, F) -> (1, G, H)
        # 3. Add positional encoding
        if len(self.blocks) > 0:
            # use positional encoding only when there are transformer blocks
            encoded_features = encoded_features + self.pos_emb # (1, G, H), (1, G, H) -> (1, G, H)
        # 4. Add transformer
        for block in self.blocks:
            encoded_features = block(encoded_features)
        # 5. Decode the groups
        group_e_cat = self.to_group_e_cat(encoded_features) # (1, G, H) -> (1, G, E)
        group_p_cat = self.to_group_p_cat(encoded_features) # (1, G, H) -> (1, G, P)
        if self.clamp_physical:
            # clamp to avoid numerical issues
            group_e_cat = torch.clamp(group_e_cat, min=-1.0, max=1.0) # (1, G, E)
            group_p_cat = torch.clamp(group_p_cat, min=-1.0, max=1.0) # (1, G, P)
        group_e_cat = group_e_cat.squeeze(0) # (1, G, E) -> (G, E)
        group_p_cat = group_p_cat.squeeze(0) # (1, G, P) -> (G, P)
        # 6. Assign center features to the points based on the nearest center index
        # Each group has the same elasticity and plasticity as the center.
        nearest_center_idx = nearest_center_idx.squeeze(0) # (1, N) -> (N)
        e_cat = group_e_cat[nearest_center_idx] # (G, E), (N) -> (N, E)
        p_cat = group_p_cat[nearest_center_idx] # (G, P), (N) -> (N, P)
        return e_cat, p_cat

def get_pos_emb(num_groups: int, hidden_size: int) -> Tensor:
    '''
    Get absolute positional embedding
    input:
        num_groups: int, the number of groups, i.e., the number of positions
        hidden_size: int, the dimension of the embedding
    output:
        position: Tensor, (G, H), the positional embedding
    '''
    assert hidden_size % 2 == 0, 'The hidden size must be even.'
    pos_emb = torch.zeros(num_groups, hidden_size)
    position = torch.arange(num_groups).unsqueeze(1).float() 
    div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_size))
    pos_emb[:, 0::2] = torch.sin(position * div_term)
    pos_emb[:, 1::2] = torch.cos(position * div_term)
    return pos_emb

class Grouper(Module):
    '''
    Grouping the points based on the centers with FPS and KNN
    input:
        x: Tensor, (B, N, 3), the coordinates of the points
    output:
        neighbors: Tensor, (B, G, M, F), the features of the neighbors
        centers: Tensor, (B, G, 3), the coordinates of the centers
    '''
    def __init__(
        self,
        num_groups: int,
        group_size: int,
        export_path: Optional[str] = None,
    ) -> None:
        super(Grouper, self).__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.export_path = export_path
        
    def forward(self, x: Tensor, features: Tensor) -> Tuple[Tensor, Tensor]:
        # 1. Select the centers with FPS
        centers = fps(x, self.num_groups) # (B, N, 3) -> (B, G, 3)
        # 2. Select the neighbors with KNN
        neighbors, nearest_center_idx = knn(x, centers, features, self.group_size) # (B, N, 3), (B, G, 3) -> (B, G, M, 3)
        if self.export_path:
            # save fps centers to .ply file for visualization
            particle_position_tensor_to_ply(centers.squeeze(0), os.path.join(self.export_path, 'fps_centers.ply'))
            print(f'FPS centers saved to {os.path.join(self.export_path, "fps_centers.ply")}')
        return neighbors, nearest_center_idx

def fps(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    fps_idx = centroids.squeeze(0) # (1, npoint) -> (npoint)
    fps_data = xyz[0, fps_idx].unsqueeze(0) # (1, N, 3) -> (1, npoint, 3)
    return fps_data

def knn(x: Tensor, centers: Tensor, features: Tensor, group_size: int) -> Tensor:
    '''
    Select the neighbors with KNN
    input:
        x: Tensor, (B, N, 3), the coordinates of the points
        centers: Tensor, (B, G, 3), the coordinates of the centers
        group_size: int, the number of neighbors
    output:
        neighbors: Tensor, (B, G, M, F), the features of the neighbors
    '''
    
    distances = square_distance(centers, x) # (B, G, 3), (B, N, 3) -> (B, G, N)
    _, neighbors_idx = torch.topk(distances, group_size, dim=-1, largest=False, sorted=False) # (B, G, N) -> (B, G, M)
    # don't need idx_base since batch size is 1. TODO: add batch size support
    neighbors_idx = neighbors_idx.reshape(-1) # (B, G, M) -> (B * G * M)
    neighbors = features.reshape(-1, features.shape[-1])[neighbors_idx] # (B * N, F) -> (B * G * M, F)
    neighbors = neighbors.reshape(1, -1, group_size, features.shape[-1]) # (B * G * M, F) -> (B, G, M, F)
    
    nearest_center_idx = torch.argmin(distances.permute(0, 2, 1), dim=-1) # (B, N, G) -> (B, N)
    
    return neighbors, nearest_center_idx
    
def square_distance(src, dst):
        '''
        Calculate the square distance between the src and dst
        input:
            src: Tensor, (B, N, 3), the coordinates of the source points
            dst: Tensor, (B, M, 3), the coordinates of the destination points
        output:
            dist: Tensor, (B, N, M), the square distance between the src and dst
        '''
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(src.shape[0], src.shape[1], 1)
        dist += torch.sum(dst ** 2, -1).view(dst.shape[0], 1, dst.shape[1])
        return dist
        
    
class GroupEncoder(Module):
    '''
    Encode the groups
    input:
        groups: Tensor, (B, G, M, F), the coordinates of the neighbors of the centers
    output:
        features: Tensor, (B, G, H), the features of the centers
    '''
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 128,
    ) -> None:
        super(GroupEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim * 4, hidden_dim * 4, 1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 4, out_dim, 1),
        )
        
    def forward(self, group_features: Tensor) -> Tensor:
        B, G, M, F = group_features.shape
        group_features = group_features.reshape(B * G, F, M) # (B, G, M, F) -> (B * G, F, M)
        group_features = self.conv1(group_features) # (B * G, F, M) -> (B * G, 2 * H, M)
        group_features_global_max = torch.max(group_features, dim=-1, keepdim=True)[0] # (B * G, 2 * H, M) -> (B * G, 2 * H, 1)
        group_features_global_max = group_features_global_max.expand(-1, -1, M) # (B * G, 2 * H, 1) -> (B * G, 2 * H, M)
        group_features = torch.cat([group_features, group_features_global_max], dim=1) # (B * G, 2 * H, M) -> (B * G, 4 * H, M)
        group_features = self.conv2(group_features) # (B * G, 4 * H, M) -> (B * G, O, M)
        group_features = torch.max(group_features, dim=-1)[0] # (B * G, O, M) -> (B * G, O)
        group_features = group_features.reshape(B, G, -1) # (B * G, O) -> (B, G, O)
        return group_features
    
class Attention(Module):
    '''
    Attention mechanism
    input:
        x: Tensor, (B, N, F), the features of the points
    output:
        x: Tensor, (B, N, F), the updated features of the points
    '''
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        B, N, F = x.shape
        qkv = self.qkv(x) # (B, N, F) -> (B, N, 3 * F)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # (B, N, 3 * F) -> (B, N, 3, H, D) -> (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2] # (3, B, H, N, D) -> (B, H, N, D)
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, H, N, D), (B, H, D, N) -> (B, H, N, N)
        attn = attn.softmax(dim=-1) # (B, H, N, N)
        attn = self.dropout(attn) # (B, H, N, N)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1) # (B, H, N, N), (B, H, N, D) -> (B, N, H * D) = (B, N, F)
        x = self.fc(x) # (B, N, H * D) -> (B, N, F)
        return x
        
class Block(Module):
    '''
    Transformer block
    input:
        x: Tensor, (B, N, F), the features of the points
    output:
        x: Tensor, (B, N, F), the updated features of the points
    '''
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x)) # (B, N, F)
        x = x + self.mlp(self.norm2(x)) # (B, N, F)
        return x
        