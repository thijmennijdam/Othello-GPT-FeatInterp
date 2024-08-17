import argparse
import os
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch as t
import numpy as np
from src.utils.mech_interp_othello_utils_adopted import (
    int_to_label, 
    OthelloBoardState, 
    to_string
)
from tqdm import tqdm
from src.othello_dataset import OthelloDataset

# Parse arguments
parser = argparse.ArgumentParser(description='Othello GPT Model Evaluation')
parser.add_argument('--num_games', type=int, default=10000, help='Number of games to evaluate')
parser.add_argument('--load_from_disk', action='store_true', help="Flag to load the dataset from disk.")
args = parser.parse_args()

n_layers = 6
d_model = 128

device = t.device("cuda" if t.cuda.is_available() else "cpu")

model_cfg = HookedTransformerConfig(
    n_layers=n_layers,
    d_model=d_model,
    d_head=64,
    n_heads=8,
    d_mlp=d_model * 4,
    d_vocab=61,
    n_ctx=59,
    act_fn="gelu",
    normalization_type="LN",
    device=device,
)

print("Loading model...")
model = HookedTransformer(model_cfg).to(device)
model.load_and_process_state_dict(t.load("checkpoints/othello/othello_gpt_6_128_lr0.001_bs512_epochs5_LN.pth"), fold_ln=False)

print("Loading data...")
dataset_path = f"data/othello_games/validation/seq_len=60/num_games={args.num_games}"

if args.load_from_disk:
    dataset = OthelloDataset.load(dataset_path, seq_len=60, max_samples=args.num_games, load_processed=True)
else:
    dataset = OthelloDataset(split='validation', seq_len=60, max_samples=args.num_games, process=True)
    dataset.save(dataset_path)

focus_games_int = dataset.games_int
focus_valid_moves = dataset.valid_moves

print("Running model to get predictions...")

focus_logits, focus_cache = model.run_with_cache(focus_games_int[:, :-1].to(device))
focus_preds = focus_logits.argmax(-1)  # shape: (num_games, 59)

correct_predictions = 0
total_predictions = 0

print("Evaluating model predictions...")
for game in tqdm(range(args.num_games), desc="Evaluating Games"):
    for move in range(focus_valid_moves.shape[1] - 1):  # Iterate over all moves
        pred = int_to_label(focus_preds[game, move])
        pred = to_string(pred)
        if focus_valid_moves[game, move, pred] == 1:
            correct_predictions += 1
        total_predictions += 1

accuracy = correct_predictions / total_predictions
print(f"The model's top-1 accuracy of legal moves over the first {args.num_games} games and all moves is: {accuracy * 100:.2f}%")