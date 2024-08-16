# Integrating your custom Othello-GPT in `transformer_lens`

The first thing you need to do is upload your trained Othello-GPT model to Hugging Face, as this is where all official transformer models are loaded from. Afterward, you need to add your model to the `transformer_lens` library, which can be done in just 4 lines. I will first explain how to upload the model to Hugging Face and then how to add it to the `transformer_lens` library.

## Uploading Your Model to Hugging Face

Go to [Hugging Face](https://huggingface.co/new) and add a public model. Then upload the `.pth` file of your Othello-GPT model that was saved in your local directory, and rename it to `final.pth` on Hugging Face. Then, add a `config.json` file. This will be used to load the model with the correct configurations in `transformer_lens`. Below is an example configuration for my model:

```json
{
    "n_layers": 6,
    "d_model": 128,
    "d_mlp": 512,
    "d_head": 64,
    "n_heads": 8,
    "n_ctx": 59,
    "d_vocab": 61,
    "act_fn": "gelu",
    "attn_only": false,
    "normalization_type": "LN"
}
```

Adjust the parameter values to match your own model's specifications. You can view my model here: [Thijmen/Othello-GPT](https://huggingface.co/Thijmen/Othello-GPT).

## Adding Your Model to `transformer_lens`

To integrate your custom model into the `transformer_lens` framework, follow these steps:

1. **Update the `OFFICIAL_MODEL_LIST`:**  
   Add the name of your model from Hugging Face to the `OFFICIAL_MODEL_LIST`. For instance, if your model is named `Thijmen/Othello-GPT`, include it in the list.

   ```python
   OFFICIAL_MODEL_LIST = [
       # existing models...
       "Thijmen/Othello-GPT",  # Add your model here
   ]
   ```

2. **Define a model alias:**  
   In the `MODEL_ALIASES` section, create an alias for your model. For example:

   ```python
   MODEL_ALIASES = {
       # existing aliases...
       "Thijmen/Othello-GPT": ["my-own-othello-model"],  # Add your alias here
   }
   ```

3. **Ensure correct model configuration loading:**  
   Your model configuration should be loaded similarly to the Baidicoot model (another Othello-GPT model). This requires modifying the `loading_from_pretrained.py` file by adding the following condition in two places: `get_pretrained_state_dict()` and `get_pretrained_model_config()`.

   Add the following line:

   ```python
   or official_model_name.startswith("Thijmen")  # Ensure your model is included
   ```

   This should be added alongside the existing check for Baidicoot models:

   ```python
   if official_model_name.startswith("Baidicoot")
   or official_model_name.startswith("Thijmen"):  # Add this line
   ```