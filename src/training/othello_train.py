import torch as t
from src.othello_gpt import create_othello_gpt
from src.othello_dataset import OthelloDataset
from src.training.config.othello_train_config import get_othello_train_config
from transformer_lens.train import train

def train_othello_gpt(d_model, n_layers, lr, batch_size, num_epochs, wandb, load_from_disk=False, dataset_path="data/othello_dataset2", max_samples=2_000_000):
    """
    Trains the Othello GPT model.

    Parameters:
        d_model (int): dimension of the model
        n_layers (int): number of layers in the model
        lr (float): learning rate
        batch_size (int): batch size
        num_epochs (int): number of epochs
        wandb (bool): flag to use Weights and Biases for logging
        load_from_disk (bool): flag to load dataset from disk
        dataset_path (str): path to save/load the dataset
        max_samples (int): maximum number of samples in the dataset

    Returns:
        None
    """
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # create the model
    model = create_othello_gpt(d_model, n_layers, device)
    
    # get training configuration
    train_cfg = get_othello_train_config(lr, batch_size, num_epochs, device, wandb)
    
    # load or create dataset
    if load_from_disk:
        custom_dataset = OthelloDataset.load(dataset_path, seq_len=59, max_samples=max_samples)
    else:
        custom_dataset = OthelloDataset(split='train', seq_len=59, max_samples=max_samples)
    
    # train the model
    train(model, train_cfg, custom_dataset)
    
    # save the model
    t.save(model.state_dict(), f"checkpoints/othello/othello_gpt_{n_layers}_{d_model}_lr{lr}_bs{batch_size}_epochs{num_epochs}_LN.pth")
