import torch
import numpy as np
import torch.nn as nn
from parse import args
from eval import evaluate_recall, evaluate_ndcg


class FedRecClient(nn.Module):
    def __init__(self, train_ind, test_ind, m_item, dim):
        super().__init__()
        self._train_ = train_ind
        self._test_ = test_ind
        self.m_item = m_item

        self._train = torch.Tensor(train_ind).long()
        self._test = torch.Tensor(test_ind).long()

        train_neg_ind = []
        for _ in train_ind:
            neg_item = np.random.randint(m_item)
            while neg_item in train_ind:
                neg_item = np.random.randint(m_item)
            train_neg_ind.append(neg_item)

        train_all = train_ind + train_neg_ind
        train_all.sort()
        self.train_all = torch.Tensor(train_all).long()

        d = {idx: train_all_idx for train_all_idx, idx in enumerate(train_all)}
        self._train_pos = torch.Tensor([d[i] for i in train_ind]).long()
        self._train_neg = torch.Tensor([d[i] for i in train_neg_ind]).long()

        self.dim = dim
        self.items_emb_grad = None
        self._user_emb = nn.Embedding(1, dim)
        nn.init.normal_(self._user_emb.weight, std=0.01)

    def forward(self, items_emb):
        scores = torch.sum(self._user_emb.weight * items_emb, dim=-1)
        return scores

    def train_(self, items_emb, reg=0.1):
        items_emb = items_emb[self.train_all].clone().detach().requires_grad_(True)
        self._user_emb.zero_grad()

        pos_items_emb = items_emb[self._train_pos]
        neg_items_emb = items_emb[self._train_neg]
        pos_scores = self.forward(pos_items_emb)
        neg_scores = self.forward(neg_items_emb)
        loss = -(pos_scores - neg_scores).sigmoid().log().sum() + \
            0.5 * (self._user_emb.weight.norm(2).pow(2) + items_emb.norm(2).pow(2)) * reg
        loss.backward()

        user_emb_grad = self._user_emb.weight.grad
        self._user_emb.weight.data.add_(user_emb_grad, alpha=-args.lr)
        self.items_emb_grad = items_emb.grad

        return self.train_all, self.items_emb_grad, loss.cpu().item()

    def eval_(self, items_emb):
        rating = self.forward(items_emb)
        rating[self._train_] = -1e9

        test_indices = self._test_
        hr_at_10 = evaluate_recall(rating, test_indices, 10)
        ndcg_at_10 = evaluate_ndcg(rating, test_indices, 10)

        return hr_at_10, ndcg_at_10
