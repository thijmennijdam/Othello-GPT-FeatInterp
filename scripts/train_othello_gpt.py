import argparse
from src.training.othello_train import train_othello_gpt

def main():
    """
    Main function to parse command line arguments and train the Othello GPT model.
    """
    parser = argparse.ArgumentParser(description="Train Othello GPT Model")
    parser.add_argument('--d_model', type=int, required=True, help='dimension of the model')
    parser.add_argument('--n_layers', type=int, required=True, help='number of layers in the model')
    parser.add_argument('--lr', type=float, required=True, help='learning rate')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--load_from_disk', action='store_true', help='load data from disk (Hugging Face Dataset)')
    parser.add_argument('--wandb', action='store_true', help='use Weights and Biases for logging')

    args = parser.parse_args()

    train_othello_gpt(
        d_model=args.d_model,
        n_layers=args.n_layers,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        load_from_disk=args.load_from_disk,
        wandb=args.wandb,
    )

if __name__ == "__main__":
    main()
