import torch as t
from sae_lens import SAETrainingRunner
from src.training.config.sae_train_config import get_sae_config

device = t.device("cuda" if t.cuda.is_available() else "cpu")

def train_sae(total_training_steps, batch_size, layer, expansion_factor, l1_coefficient, lr):
    """
    Trains the Sparse Autoencoder (SAE) model.

    Parameters:
        total_training_steps (int): total number of training steps
        batch_size (int): batch size
        layer (int): specific layer to apply the SAE
        expansion_factor (int): factor to expand the width of the SAE
        l1_coefficient (float): coefficient to control sparsity through L1 regularization
        lr (float): learning rate

    Returns:
        None
    """
    cfg = get_sae_config(total_training_steps, batch_size, layer, expansion_factor, l1_coefficient, lr)
    sparse_autoencoder = SAETrainingRunner(cfg).run()
    sparse_autoencoder.save_model(f"checkpoints/sae/{cfg.model_name}_{cfg.hook_name}_{cfg.hook_layer}_{cfg.expansion_factor}_{cfg.l1_coefficient}_{cfg.train_batch_size_tokens}_{cfg.training_tokens}_{cfg.lr}_{cfg.l1_coefficient}")
