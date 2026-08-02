import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from parse import args


class FedRecServer(nn.Module):
    def __init__(self, m_item, dim, agg_type='Sum'):
        super().__init__()
        self.m_item = m_item
        self.dim = dim
        self.items_emb = nn.Embedding(m_item, dim)
        self.agg_type = agg_type
        nn.init.normal_(self.items_emb.weight, std=0.01)

        # InertiaDamp defense state
        self.last_round_grads = torch.zeros(m_item, dim).to(args.device)
        self.relative_ratio = 1.2
        self.damping_factor = 0.1

    def train_(self, clients, batch_clients_idx):
        batch_loss = []
        collected_grads = {}

        for idx in batch_clients_idx:
            client = clients[idx]
            items, items_emb_grad, loss = client.train_(self.items_emb.weight)

            with torch.no_grad():
                norm = items_emb_grad.norm(2, dim=-1, keepdim=True)
                too_large = norm[:, 0] > args.grad_limit
                items_emb_grad[too_large] /= (norm[too_large] / args.grad_limit)

                items_list = items.tolist()
                for i, item_id in enumerate(items_list):
                    if item_id not in collected_grads:
                        collected_grads[item_id] = []
                    collected_grads[item_id].append(items_emb_grad[i])

            if loss is not None:
                batch_loss.append(loss)

        new_batch_grad = torch.zeros_like(self.items_emb.weight)

        with torch.no_grad():
            if self.agg_type == 'InertiaDamp':
                item_sim_stats = {}

                for item_id, grads in collected_grads.items():
                    current_sum_grad = torch.sum(torch.stack(grads), dim=0)
                    prev_grad = self.last_round_grads[item_id]

                    if prev_grad.norm() > 1e-9 and current_sum_grad.norm() > 1e-9:
                        sim = F.cosine_similarity(current_sum_grad.unsqueeze(0),
                                                  prev_grad.unsqueeze(0)).item()
                        item_sim_stats[item_id] = (sim, current_sum_grad)
                    else:
                        item_sim_stats[item_id] = (0.0, current_sum_grad)

                active_sims = [v[0] for v in item_sim_stats.values() if v[0] != 0]
                avg_batch_sim = np.mean(active_sims) if active_sims else 0
                threshold = avg_batch_sim * self.relative_ratio

                for item_id, (sim, current_sum_grad) in item_sim_stats.items():
                    if sim > threshold and sim > 0.1:
                        new_batch_grad[item_id] = current_sum_grad * self.damping_factor
                    else:
                        new_batch_grad[item_id] = current_sum_grad

                    self.last_round_grads[item_id] = current_sum_grad

            else:  # Sum (default, no defense)
                for item_id, grads in collected_grads.items():
                    new_batch_grad[item_id] = torch.sum(torch.stack(grads), dim=0)

        with torch.no_grad():
            self.items_emb.weight.data.add_(new_batch_grad, alpha=-args.lr)

        return batch_loss

    def eval_(self, clients):
        test_cnt, total_hr, total_ndcg = 0, 0.0, 0.0
        for client in clients:
            test_result = client.eval_(self.items_emb.weight)
            if test_result is not None:
                hr, ndcg = test_result
                test_cnt += 1
                total_hr += hr
                total_ndcg += ndcg
        return (total_hr / test_cnt, total_ndcg / test_cnt) if test_cnt > 0 else (None, None)
