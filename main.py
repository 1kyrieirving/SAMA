import torch
import random
import numpy as np
from time import time
from parse import args
from data import load_dataset

from FedRec.server import FedRecServer
from FedRec.client import FedRecClient
from FedRec.malicious_client import MaliciousFedRecClient


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args_str = ",".join([f"{k}={v}" for k, v in args.__dict__.items()])
    print(f"Arguments: {args_str}")

    t0 = time()
    m_item, all_train_ind, all_test_ind = load_dataset(args.path + args.dataset)

    server = FedRecServer(m_item, args.dim, agg_type=args.agg_type).to(args.device)

    num_users = len(all_train_ind)
    num_malicious = max(1, int(num_users * 0.05))
    malicious_idx_set = set(np.random.choice(num_users, num_malicious, replace=False))

    clients = []
    for i in range(num_users):
        train_ind = all_train_ind[i]
        test_ind = all_test_ind[i]

        if i in malicious_idx_set:
            client = MaliciousFedRecClient(
                train_ind, test_ind, m_item, args.dim, args.device,
                attack_strength=3.0, alpha=0.5, top_ratio=0.2
            ).to(args.device)
        else:
            client = FedRecClient(train_ind, test_ind, m_item, args.dim).to(args.device)

        clients.append(client)

    print(f"Load data done [{time() - t0:.1f}s]. "
          f"#Benign={len(clients) - num_malicious}, #Malicious={num_malicious}, #Items={m_item}")

    with torch.no_grad():
        test_result = server.eval_(clients)
    print(f"Iteration 0(init), HR@10={test_result[0]:.4f}, NDCG@10={test_result[1]:.4f}")

    try:
        for epoch in range(1, args.epochs + 1):
            t1 = time()

            rand_clients = np.arange(len(clients))
            np.random.shuffle(rand_clients)

            total_loss = []
            for i in range(0, len(rand_clients), args.batch_size):
                batch_clients_idx = rand_clients[i: i + args.batch_size]
                loss = server.train_(clients, batch_clients_idx)
                total_loss.extend(loss)
            avg_loss = np.mean(total_loss).item()

            t2 = time()
            with torch.no_grad():
                hr, ndcg = server.eval_(clients)

            print(f"Iteration {epoch}, "
                  f"loss = {avg_loss:.5f} [{t2 - t1:.1f}s], "
                  f"HR@10 = {hr:.4f}, NDCG@10 = {ndcg:.4f}")
    except KeyboardInterrupt:
        print("Training interrupted.")


if __name__ == "__main__":
    setup_seed(20211111)
    main()
