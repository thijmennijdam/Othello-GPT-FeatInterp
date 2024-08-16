from transformer_lens import HookedTransformer, HookedTransformerConfig

def get_othello_gpt_config(d_model, n_layers, device):
    return HookedTransformerConfig(
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

def create_othello_gpt(d_model, n_layers, device):
    model_cfg = get_othello_gpt_config(d_model, n_layers, device)
    return HookedTransformer(model_cfg).to(device)