import torch
import torch.nn.functional as F
from losses.grl import GradientReverseLayer  #

class SAM:

    def __init__(self,
                 num_classes: int,
                 device: torch.device = "cuda",
                 tau: float = 0.02,
                 cnt_pow: float = -0.1,
                 smooth_alpha: float = 0.05,
                 ema_momentum: float = 0.3,
                 init_loss: float = 1,
                 eps: float = 1e-6):
        self.C = num_classes
        self.dev = device

        self.tau   = tau
        self.alpha = smooth_alpha
        self.cnt_pow = cnt_pow
        self.eps  = eps

        #  per-class EMA loss
        self.ema_m  = ema_momentum
        self.ema_loss = torch.full((num_classes,),
                                   init_loss,
                                   dtype=torch.float32,
                                   device=device)

    # -------------------------------------------------
    # main interface
    @torch.no_grad()
    def __call__(self,
                 local_loss: torch.Tensor,   # [C]
                 class_cnt: torch.Tensor     # [C]
                 ) -> torch.Tensor:          # weights [C]

        C = self.C
        assert local_loss.shape[0] == C and class_cnt.shape[0] == C

        valid = class_cnt > 0
        empty = ~valid

        loss_smoothed = torch.zeros_like(local_loss)
        loss_smoothed[valid] = (1-self.ema_m) * local_loss[valid] + self.ema_m * self.ema_loss[valid]
        loss_smoothed[empty] = self.ema_loss[empty]

        diff = (0.69 - loss_smoothed).clamp(min=0.)
        pos_mask      = diff > 0
        aligned_mask  = diff == 0

        soft_pos = torch.softmax(diff[pos_mask] / self.tau, dim=0)
        U        = 1.0 / C

        base_w = torch.zeros(C, device=self.dev)

        base_w[pos_mask] = (1 - self.alpha) * soft_pos + self.alpha * U

        base_w[aligned_mask] = self.alpha * U

        if self.cnt_pow != 0:
            scale = class_cnt.float().clamp(min=1.).pow(self.cnt_pow)
            base_w[valid] *= scale[valid]          # empty, scale=1

        weights = base_w / base_w.sum().clamp(min=self.eps)

        self._update_ema(valid, local_loss)
        return weights

    # -------------------------------------------------
    # EMA update
    # -------------------------------------------------
    @torch.no_grad()
    def _update_ema(self, valid: torch.Tensor, local_loss: torch.Tensor):
        self.ema_loss[valid] = (self.ema_m * self.ema_loss[valid]
                               + (1 - self.ema_m) * local_loss[valid])

def compute_focal_domain_loss(shared_feats_src, shared_feats_tgt,
                              src_labels, tgt_pseudo_labels,
                              model, subdomain_attention):

    device = shared_feats_src.device
    C = model.num_classes

    # -------------------------------------  per-class  & loss
    class_counts        = torch.zeros(C, device=device)
    local_loss_per_cls  = torch.zeros(C, device=device)

    rev_src = GradientReverseLayer.apply(shared_feats_src, 1.0)
    rev_tgt = GradientReverseLayer.apply(shared_feats_tgt, 1.0)

    for c in range(C):
        src_mask = (src_labels == c)
        tgt_mask = (tgt_pseudo_labels == c)

        src_cnt, tgt_cnt = int(src_mask.sum()), int(tgt_mask.sum())
        if src_cnt == 0 or tgt_cnt == 0:
            continue   # 本批无样本，loss 保持 0

        feats_c = torch.cat([rev_src[src_mask], rev_tgt[tgt_mask]], 0)
        dom_lbl = torch.cat([torch.zeros(src_cnt, device=device),
                             torch.ones(tgt_cnt, device=device)], 0)

        dom_logit = model.forward_local_discriminator(feats_c, c).squeeze(1)
        loss_c = F.binary_cross_entropy_with_logits(dom_logit, dom_lbl.float())

        class_counts[c]       = src_cnt + tgt_cnt
        local_loss_per_cls[c] = loss_c

    # --------  dyn_w  --------
    # input subdomain_attention
    weights = subdomain_attention(local_loss_per_cls, class_counts)
    weighted_local_loss = (local_loss_per_cls * weights).sum()

    return weighted_local_loss, weights.detach().cpu(), local_loss_per_cls