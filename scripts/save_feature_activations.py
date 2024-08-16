import argparse
import os
import torch as t
from transformer_lens import HookedTransformer, utils
from tqdm import tqdm
import einops
import torch.nn.functional as F
import re
from sae_lens import SAE
from src.othello_dataset import OthelloDataset

device = t.device("cuda" if t.cuda.is_available() else "cpu")
    
def compute_feat_acts(model_acts, feature_idx: list[int], sae):
    """
    compute the feature activations of the sparse autoencoder (sae).

    parameters:
        model_acts (tensor): activations from the model.
        feature_idx (list[int]): indices of the features to compute.
        sae (sae): the sparse autoencoder model.

    returns:
        tensor: feature activations.
    """
    feature_act_dir = sae.W_enc[:, feature_idx]
    feature_bias = sae.b_enc[feature_idx]
    x_cent = model_acts - sae.b_dec
    feat_acts_pre = einops.einsum(
        x_cent, feature_act_dir, "batch seq d_in, d_in feats -> batch seq feats"
    )
    feat_acts = F.relu(feat_acts_pre + feature_bias)
    return feat_acts

def get_feature_data(
    sae,
    model,
    tokens,
    feature_indices,
    minibatch_size_tokens,
    hook_point,
    expansion_factor,
    l1_penalty
):
    """
    extract and save feature activations for a specific set of features.

    parameters:
        sae (sae): the sparse autoencoder model.
        model (hookedtransformer): the transformer model.
        tokens (tensor): the input tokens.
        feature_indices (int or list[int]): indices of features to extract.
        minibatch_size_tokens (int): size of each minibatch in tokens.
        hook_point (str): hook point in the model to extract activations.
        expansion_factor (int): expansion factor for the sae.
        l1_penalty (float): l1 regularization penalty.
    """
    if isinstance(feature_indices, int):
        feature_indices = [feature_indices]

    # split tokens into minibatches and move them to the device
    token_minibatches = tokens.split(minibatch_size_tokens)
    token_minibatches = [tok.to(device) for tok in token_minibatches]

    all_feat_acts = []

    def hook_fn_store_act(activation, hook):
        hook.ctx["activation"] = activation

    for minibatch_idx, minibatch in enumerate(token_minibatches):
        
        # extract the layer number from the hook point
        layer_match = re.match(r"blocks\.(\d+)\.", hook_point)
        assert layer_match, f"error: expecting hook_point to be 'blocks.{{layer}}.{{...}}', but got {hook_point!r}"
        hook_layer = int(layer_match.group(1))
        hook_point_resid_final = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
        
        with t.no_grad():
            # run the model and store the activations at the specified hook points
            output = model.run_with_hooks(
                minibatch,
                fwd_hooks=[
                    (hook_point, hook_fn_store_act),
                    (hook_point_resid_final, hook_fn_store_act),
                ],
            )
            model_acts = model.hook_dict[hook_point].ctx.pop("activation")
        
        # print(f"After retrieving activations: {t.cuda.memory_allocated(device) / 1024**2:.2f} MB allocated, {t.cuda.memory_reserved(device) / 1024**2:.2f} MB reserved")
        # compute feature activations using the model activations
        feat_acts = compute_feat_acts(
            model_acts=model_acts.to(device),
            feature_idx=feature_indices,
            sae=sae
        )
        all_feat_acts.append(feat_acts)
    
    all_feat_acts = t.cat(all_feat_acts, dim=0)

    # create the directory to save the activations if it doesn't exist
    output_dir = f"data/saved_feature_activations/{hook_point}/{expansion_factor}/{l1_penalty}"
    os.makedirs(output_dir, exist_ok=True)

    # save the feature activations to a file
    t.save(all_feat_acts, f"{output_dir}/{feature_indices[0]}-{feature_indices[-1]}.pt")


if __name__ == "__main__":
    # argument parser for command-line options
    parser = argparse.ArgumentParser(description="Save feature activations from Othello-GPT using trained SAEs.")
    
    parser.add_argument('--layers', nargs='+', type=int, required=True, help="Layers of the model to extract activations from.")
    parser.add_argument('--expansion_factors', nargs='+', type=int, required=True, help="Expansion factors for the SAEs.")
    parser.add_argument('--l1_coefficient', type=float, required=True, help="L1 regularization coefficient used in training the SAEs.")
    parser.add_argument('--num_games', type=int, required=True, help="Number of games to process in each batch.")
    parser.add_argument('--load_from_disk', action='store_true', help="Flag to load the dataset from disk.")
    
    args = parser.parse_args()
    
    # load the pre-trained othello-gpt model from hugging face
    model = HookedTransformer.from_pretrained("Thijmen/Othello-GPT").to(device)
    
    # load the dataset
    dataset_path = f"data/othello_games/validation/seq_len=59/num_games={args.num_games}"

    if args.load_from_disk:
        dataset = OthelloDataset.load(path=dataset_path, seq_len=59, max_samples=args.num_games)
    else:
        dataset = OthelloDataset(split='validation', seq_len=59, max_samples=args.num_games)
        dataset.save(dataset_path)
        
    # iterate over each sae layer and expansion factor
    for sae_layer in args.layers:
        for sae_expansion_factor in args.expansion_factors:
            # load the pre-trained sae model
            model_name = (
                f"checkpoints/sae/my-own-othello-model_blocks.{sae_layer}.hook_resid_pre_"
                f"{sae_layer}_{sae_expansion_factor}_{args.l1_coefficient}_512_102400000_0.0001_{args.l1_coefficient}"
            )
            sae = SAE.load_from_pretrained(model_name).to(device)

            # define the step size for feature index batching
            d_sae = sae.cfg.d_sae
            step = 64

            # create a list of feature indices to process in batches
            feature_idx_list = [range(i, i + step) for i in range(0, d_sae, step)]
            sae.eval()

            # process and save feature activations for each batch of features
            for feature_indices in tqdm(feature_idx_list):
                hook_name = sae.cfg.hook_name

                get_feature_data(
                    sae=sae,
                    model=model,
                    tokens=dataset.games_int,
                    feature_indices=feature_indices,
                    minibatch_size_tokens=512,
                    hook_point=hook_name,
                    expansion_factor=sae_expansion_factor,
                    l1_penalty=args.l1_coefficient
                )