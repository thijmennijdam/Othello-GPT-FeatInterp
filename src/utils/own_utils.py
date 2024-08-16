import pandas as pd
import os
import numpy as np

def write_csv(sae_layer,  sea_expansion_factor, n_features_active, n_mine, n_theirs, n_blank, moves_per_feature, avg_seq_lens, exp_dir):

    n_mine, std_n_mine = np.mean(n_mine), np.std(n_mine)
    n_theirs, std_n_theirs = np.mean(n_theirs), np.std(n_theirs)
    n_blank, std_n_blank = np.mean(n_blank), np.std(n_blank)
    moves_per_feature, std_moves_per_feature = np.mean(moves_per_feature), np.std(moves_per_feature)
    avg_seq_lens, std_avg_seq_lens = np.mean(avg_seq_lens), np.std(avg_seq_lens)
    
    # Define the path to the CSV file
    csv_file_path = f"{exp_dir}/results.csv"

    # Data to write as a dictionary
    data = {
        'features_active': [n_features_active],
        'mine_board_states': [n_mine],
        'std_mine_board_states': [std_n_mine],
        'their_board_states': [n_theirs],
        'std_their_board_states': [std_n_theirs],
        'blank_board_states': [n_blank],
        'std_blank_board_states': [std_n_blank],
        'average_moves_per_feature': [moves_per_feature],
        'std_average_moves_per_feature': [std_moves_per_feature],
        'avg_seq_lens': [avg_seq_lens],
        'std_avg_seq_lens': [std_avg_seq_lens],
    }

    # Define the row label
    row_label = f"L{sae_layer}E{sea_expansion_factor}"
    data['row_name'] = [row_label]

    # Create a DataFrame
    df_new_row = pd.DataFrame(data)

    # Check if the file exists and then either append to it or write a new file
    if os.path.exists(csv_file_path):
        # File exists, read the existing data and append the new row
        df_existing = pd.read_csv(csv_file_path)
        df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)
        df_combined.to_csv(csv_file_path, index=False)
    else:
        # File does not exist, write the new DataFrame to a new CSV file
        df_new_row.to_csv(csv_file_path, index=False)
