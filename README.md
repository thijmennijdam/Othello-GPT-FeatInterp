# How to train Othello GPT

To train the model, run the following command (example):

```bash
python -m scripts.train_othello_gpt --d_model 128 --n_layers 6 --lr 0.01 --batch_size 512 --epochs 5 --wandb
```

### Command Line Arguments

- `--d_model`: dimension of the model
- `--n_layers`: number of layers in the model
- `--lr`: learning rate
- `--batch_size`: batch size
- `--epochs`: number of epochs
- `--load_from_disk`: load data from disk
- `--wandb`: use Weights and Biases for logging
- `--load_from_disk`: load data from disk

The data at first takes some time to process (~7min depending on hardware), as the model uses a context window of 59 tokens, instead of 60, which is the length of the data. Once the data is loaded, you can use the `--load_from_disk` flag to load the preprocessed data from disk, as it is saved automatically.

The trained model will be saved in the `checkpoints/othello` directory with a filename containing the model's hyperparameters.

### Evaluate Othello-GPT

To evaluate the top-1 accuracy of legal moves for the Othello-GPT model, run the following command:

```bash
python -m scripts.eval_othello_gpt --num_games 10_000
```

This command will evaluate the model on 10,000 games, and the process should take approximately 5 minutes.

# How to train SAEs

You need to upload the OthelloGPT as a Hugging Face model and set the correct naming in the TransformerLens library. Details for this can be found in 'TransformerLens_changes.md'. The TransformerLens library can only load an official list of Hugging Face models, and you need to add it manually there. Once your model is uploaded, you can train the SAEs on this model using SAELens.

To train the SAEs, run the following command (example):

```bash
python -m scripts.train_sae --total_training_steps 200_000 --batch_size 512 --layer 5 --expansion_factor 8 --l1_coefficient 1e-2 --lr 1e-4
```

### Command Line Arguments

- `--total_training_steps`: total training steps
- `--batch_size`: batch size
- `--layer`: layer to apply the SAE on
- `--expansion_factor`: expansion factor for the SAE
- `--l1_coefficient`: L1 coefficient to control sparsity
- `--lr`: learning rate

The trained model will be saved in the `checkpoints/sae` directory with a filename containing the model's hyperparameters.

# How to Save SAE Activations

To run a batch of games through Othello-GPT and the SAEs, and cache the activations, run the following command (example):

```bash
python -m scripts.save_feature_activations --layers 1 3 5 --expansion_factors 8 16 --l1_coefficient 1e-2 --num_games 25000
```

### Command Line Arguments

- `--layers`: the layers of the model to extract activations from
- `--expansion_factors`: the expansion factors of the SAEs
- `--l1_coefficient`: the L1 regularization coefficient used in training the SAEs
- `--batch_size`: the number of games to process in each batch
- `--load_from_disk`: load data from disk

This command saves the feature activations of 6 SAEs: from layers 1, 3, and 5 with expansion factors 8 and 16, all trained with an L1 coefficient of 1e-2. A batch size of 25,000 games is used, resulting in a tensor of shape `[25000, d_model * expansion_factor]` that contains all the activations of the SAE.

The saved activations can take up significant storage space, but this approach can greatly reduce the time required when experimenting with extracting notable features from the activations.

The activations will be saved in the `data/saved_feature_activations` directory with a structure that includes the layer, expansion factor, and L1 coefficient.

# How to Extract Notable Features from Saved Feature Activations

After running the `save_feature_activations` script, you can extract notable features based on a threshold. This process is described in detail in the TODO (ref to blogpost section).

To extract notable features, run the following command (example):

```bash
python -m scripts.extract_notable_features --layers 1 3 5 --expansion_factors 8 16 --l1_coefficient 1e-2 --num_games 25000 --threshold 0.99 --k 10
```

### Command Line Arguments

- `--layers`: the layers of the model to extract features from
- `--expansion_factors`: the expansion factors of the SAEs
- `--l1_coefficient`: the L1 regularization coefficient used in training the SAEs
- `--num_games`: the number of games to process for feature extraction
- `--threshold`: the threshold used to determine notable features
- `--k`: the top-k games to extract of the notable features.
- `--load_from_disk`: load data from disk

This command will extract notable features from the saved activations in layers 1, 3, and 5 with expansion factors 8 and 16, all trained with an L1 coefficient of 1e-2. The extraction will be performed on 25,000 games, using a threshold of 0.99 to determine the notability of features. The top-k games where the features were most active will be stored as well. 

The average board state plots will be saved in the directory `data/extracted_notable_features`.
