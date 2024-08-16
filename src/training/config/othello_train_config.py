from transformer_lens.train import HookedTransformerTrainConfig

def get_othello_train_config(lr, batch_size, num_epochs, device, wandb):
    """
    Generates a training configuration for the Othello GPT model.

    Parameters:
        lr (float): Learning rate.
        batch_size (int): Batch size.
        num_epochs (int): Number of epochs.
        device (str): Device to use for training ('cpu' or 'cuda').
        wandb (bool): Flag to use Weights and Biases for logging.

    Returns:
        HookedTransformerTrainConfig: Configuration object for training.
    """
    return HookedTransformerTrainConfig(
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        optimizer_name="AdamW",
        weight_decay=0.01,
        device=device,
        wandb=wandb,
        wandb_project_name="OthelloGPTTraining",
        save_dir="othello"
    )
