import argparse
import os
import torch
import numpy as np
from src.othello_dataset import OthelloDataset
from src.utils.mech_interp_othello_utils_adopted import plot_single_board, int_to_label
from src.utils.own_utils import write_csv
from src.utils.plotly_utils import plot_square_as_board

def extract_notable_games(dataset, sae_layer, sae_expansion_factor, l1_penalty, threshold, k):    
    """
    Extract and visualize notable game sequences based on activations from a sparse autoencoder (SAE).

    Parameters:
        dataset (OthelloDataset): The dataset containing game states and moves.
        sae_layer (int): The SAE layer being analyzed.
        sae_expansion_factor (int): Expansion factor for the SAE layer.
        l1_penalty (float): L1 regularization penalty applied during SAE training.
        threshold (float): The threshold used to determine notable features.
        k (int): The top-k games to extract based on feature activation.
    """
    
    print(f"Starting extraction for SAE Layer {sae_layer}, Expansion Factor {sae_expansion_factor}, L1 Penalty {l1_penalty}, Threshold {threshold}, Top-k {k}")
    
    focus_states_flipped_value = dataset.states_flipped_value
    focus_games_int = dataset.games_int
    num_games = focus_states_flipped_value.shape[0]
    
    d_sae = 128 * sae_expansion_factor
    hook_point = f"blocks.{sae_layer}.hook_resid_pre"
    activations_dir = f"data/saved_feature_activations/{hook_point}/{sae_expansion_factor}/{l1_penalty}"

    step = 64
    feature_idx_list = [
        range(i, i+step) for i in range(0, d_sae, step)
    ]
    feature_count = 0

    n_features_active = 0
    n_theirs = []
    n_mine = []
    n_blank = []
    moves_per_feature = []
    avg_seq_lens = []
    
    print(f"Processing features in steps of {step}")
    
    # iterate over each set of feature indices
    for feature_idx in feature_idx_list:
        print(f"Loading activations for features {feature_idx[0]} to {feature_idx[-1]}")
        # load activations for the current set of features
        activations = torch.load(f"{activations_dir}/{feature_idx[0]}-{feature_idx[-1]}.pt")
        np_activations = activations.detach().cpu().numpy()
        
        # process each feature in the current set
        for feature in range(0, step):
            feature_acts = np_activations[:, :, feature]
            top_moves = torch.tensor(feature_acts > np.quantile(feature_acts, threshold)).cuda()
            
            # calculate board states at top moves 
            board_state_at_top_moves = torch.stack([
                (focus_states_flipped_value == 2)[:, :-1][top_moves].float().mean(0),
                (focus_states_flipped_value == 1)[:, :-1][top_moves].float().mean(0),
                (focus_states_flipped_value == 0)[:, :-1][top_moves].float().mean(0)
            ])
            
            n_moves = top_moves.sum().item()
            total_board_state_feature = (board_state_at_top_moves > threshold).sum().item()
            mine_board_state_feature = (board_state_at_top_moves[0] > threshold).sum().item()   
            their_board_state_feature = (board_state_at_top_moves[1] > threshold).sum().item()
            blank_board_state_feature = (board_state_at_top_moves[2] > threshold).sum().item()
            
            plots_dir = f"plots/qualitative"

            # check if the feature is notable and should be visualized
            if (mine_board_state_feature > 0 or their_board_state_feature > 0 ) and n_moves >= 100:
                n_features_active += 1
                moves_per_feature.append(n_moves)
                n_mine.append(mine_board_state_feature)
                n_theirs.append(their_board_state_feature)
                n_blank.append(blank_board_state_feature)
                
                # flatten activations for top-k processing
                activations_feature = activations[:, :, feature]
                flattened_activations = activations_feature.view(-1)

                # get the top k activations and their indices
                top_k_values, top_k_indices = torch.topk(flattened_activations, k)
                game_indices = top_k_indices // activations_feature.size(1)
                move_indices = top_k_indices % activations_feature.size(1)
                
                # create directory for storing plots
                feature_dir = f"{plots_dir}/layer={sae_layer}/expansion_factor={sae_expansion_factor}/l1_penalty={l1_penalty}/n_games={num_games}/threshold={threshold}/L{sae_layer}F{feature_count}_total_moves={n_moves}_M={mine_board_state_feature}_T={their_board_state_feature}_B={blank_board_state_feature}"
                os.makedirs(feature_dir, exist_ok=True)
                
                # generate plots for the top-k sequences
                total_seq_len = 0
                for i in range(k):
                    sequence = focus_games_int[game_indices[i], :move_indices[i] + 1]
                    seq_length = sequence.shape[0]
                    total_seq_len += seq_length
                    if seq_length <= 1:
                        continue
                    plot_single_board(int_to_label(sequence), return_fig=True).write_image(f"{feature_dir}/k={i}.png")
                
                average_seq_len = total_seq_len / k
                avg_seq_lens.append(average_seq_len)
                
                # plot the board states corresponding to the top moves
                plot_square_as_board(
                    board_state_at_top_moves,
                    filename=feature_dir, 
                    facet_col=0,
                    facet_labels=["Mine", "Theirs", "Blank"],
                    title=f"Layer {sae_layer} Feature {feature_count} - Total moves: {n_moves}"
                )
            
            feature_count += 1
        
    # write the results to a CSV file
    write_csv(sae_layer,  sae_expansion_factor, n_features_active, n_mine, n_theirs, n_blank, moves_per_feature, avg_seq_lens, plots_dir)

if __name__ == "__main__":
    # argument parser for command-line options
    parser = argparse.ArgumentParser(description="Extract notable features from saved SAE activations.")
    
    parser.add_argument('--layers', nargs='+', type=int, required=True, help="Layers of the model to extract features from.")
    parser.add_argument('--expansion_factors', nargs='+', type=int, required=True, help="Expansion factors for the SAEs.")
    parser.add_argument('--l1_coefficient', type=float, required=True, help="L1 regularization coefficient used in training the SAEs.")
    parser.add_argument('--num_games', type=int, required=True, help="Number of games to process for feature extraction.")
    parser.add_argument('--threshold', type=float, required=True, help="Threshold to determine notable features.")
    parser.add_argument('--k', type=int, required=True, help="Top-k games to extract based on feature activation.")
    parser.add_argument('--load_from_disk', action='store_true', help="Flag to load the dataset from disk.")
    
    args = parser.parse_args()

    # load or process the dataset
    dataset_path = f"data/othello_games/validation/seq_len=60/num_games={args.num_games}"
    
    if args.load_from_disk:
        print("Loading dataset")
        dataset = OthelloDataset.load(dataset_path, seq_len=60, max_samples=args.num_games, load_processed=True)
    else:
        dataset = OthelloDataset(split='validation', seq_len=60, max_samples=args.num_games, process=True)
        dataset.save(dataset_path)

    # extract and visualize notable games for each combination of sae layer and expansion factor
    for sae_layer in args.layers:
        for sae_expansion_factor in args.expansion_factors:
            print(f"Extracting notable games for layer {sae_layer} and expansion factor {sae_expansion_factor}")
            extract_notable_games(dataset, sae_layer, sae_expansion_factor, args.l1_coefficient, args.threshold, args.k)