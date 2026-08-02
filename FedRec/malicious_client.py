import torch
import torch.nn.functional as F

from FedRec.client import FedRecClient


class MaliciousFedRecClient(FedRecClient):
    """
    Sparse Adaptive Momentum Attack (SAMA).
    Locally trains to get benign gradient norm; attack uses EMA displacement to push items
    with dynamic strength control.
    """
    def __init__(self, train_ind, test_ind, m_item, dim, device,
                 attack_strength=1.5, alpha=0.5, top_ratio=0.1):
        super().__init__(train_ind, test_ind, m_item, dim)

        self.device = device
        self.attack_strength = attack_strength
        self.alpha = alpha
        self.top_ratio = top_ratio

        self.velocity_ema = None
        self.last_items_emb = None
        self.smoothed_loss = None
        self.decay_factor = 0.5
        self.growth_factor = 1.05
        self.min_strength = 1.0
        self.max_strength = 5
        self.patience_counter = 0
        self.max_patience = 2
        self.loss_ratio_threshold = 1.50

    def train_(self, items_emb):
        budget = len(self.train_all)
        _, benign_grad, loss_val = super().train_(items_emb)

        with torch.no_grad():
            benign_norm = benign_grad.norm(2, dim=-1).mean().item()
            benign_norm = max(benign_norm, 1e-6)

        # Dynamic strength adjustment
        if self.smoothed_loss is None:
            self.smoothed_loss = loss_val
        else:
            loss_ratio = loss_val / (self.smoothed_loss + 1e-6)

            if loss_ratio > self.loss_ratio_threshold:
                self.patience_counter += 1
                if self.patience_counter >= self.max_patience:
                    self.attack_strength = max(self.min_strength,
                                               self.attack_strength * self.decay_factor)
                    self.patience_counter = 0
            else:
                self.patience_counter = 0
                if loss_ratio < 1.2:
                    self.attack_strength = min(self.max_strength,
                                               self.attack_strength * self.growth_factor)

            self.smoothed_loss = 0.8 * self.smoothed_loss + 0.2 * loss_val

        # Sparse Adaptive Momentum Push
        items_emb_detach = items_emb.detach()

        if self.last_items_emb is None:
            velocity = torch.zeros_like(items_emb_detach)
        else:
            velocity = items_emb_detach - self.last_items_emb

        if self.velocity_ema is None:
            self.velocity_ema = velocity.clone()
        else:
            self.velocity_ema = self.alpha * velocity + (1 - self.alpha) * self.velocity_ema

        speed = torch.norm(velocity, dim=1)
        num_candidates = budget
        candidate_indices = torch.randperm(speed.numel(), device=speed.device)[:num_candidates]
        topk_indices = torch.sort(candidate_indices).values.to(self.device)

        selected_ema = self.velocity_ema[topk_indices]
        direction = F.normalize(selected_ema, dim=-1)
        adaptive_strength = benign_norm * self.attack_strength
        grad_topk = -direction * adaptive_strength

        self.last_items_emb = items_emb_detach.clone()

        return topk_indices, grad_topk, loss_val

    def eval_(self, items_emb):
        return None
