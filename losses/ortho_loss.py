import torch
import torch.nn.functional as F

def orthogonality_loss(features_shared, features_private, eps=1e-6):

    fs = F.normalize(features_shared, dim=1)
    fp = F.normalize(features_private, dim=1)

    cross_sim = torch.matmul(fs, fp.T)  # [N, N]
    cross_loss = torch.norm(cross_sim, p='fro')

    self_sim_shared = torch.matmul(fs, fs.T) - torch.eye(fs.size(0)).to(fs.device)
    self_sim_private = torch.matmul(fp, fp.T) - torch.eye(fp.size(0)).to(fp.device)
    self_loss = (torch.norm(self_sim_shared, p='fro') + torch.norm(self_sim_private, p='fro')) * 0.5

    loss= cross_loss + self_loss
    return loss