import argparse
import torch.cuda as cuda


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAMA: Sparse Adaptive Momentum Attack against Federated Recommender Systems"
    )
    parser.add_argument('--dim', type=int, default=32, help='Dimension of latent vectors.')
    parser.add_argument('--path', nargs='?', default='Data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-100k/', help='Dataset name.')
    parser.add_argument('--device', nargs='?', default='cuda' if cuda.is_available() else 'cpu',
                        help='Device to run the model.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--agg_type', type=str, default='Sum',
                        help='Aggregation type: Sum (no defense), SDCSum (our defense).')
    parser.add_argument('--grad_limit', type=float, default=1., help='L2-norm limit for item gradients.')
    return parser.parse_args()


args = parse_args()
