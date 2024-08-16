import argparse
from src.training.sae_train import train_sae

def main():
    """
    Main function to parse command line arguments and train the SAE model.
    """
    parser = argparse.ArgumentParser(description='Train SAE model with specified parameters.')
    parser.add_argument('--total_training_steps', type=int, required=True, help='Total training steps')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--layer', type=int, required=True, help='Layer')
    parser.add_argument('--expansion_factor', type=int, required=True, help='Expansion factor')
    parser.add_argument('--l1_coefficient', type=float, required=True, help='L1 coefficient')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')

    args = parser.parse_args()

    train_sae(
        total_training_steps=args.total_training_steps,
        batch_size=args.batch_size,
        layer=args.layer,
        expansion_factor=args.expansion_factor,
        l1_coefficient=args.l1_coefficient,
        lr=args.lr
    )

if __name__ == "__main__":
    main()