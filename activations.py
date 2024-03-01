import torch
from jaxtyping import Int, Float
import sys
from transformers import BertTokenizer, BertModel

sys.path.append('../')
from models import get_model_from_name


def get_layers_to_enumerate(model):
    model_name = model.config._name_or_path
    if 'gpt' in model_name:
        return model.transformer.h
    elif 'pythia' in model_name:
        return model.gpt_neox.layers
    elif 'bert' in model_name:
        return model.encoder.layer
    elif 'Mistral' in model_name:
        return model.model.layers
    else:
        raise ValueError(f"Unsupported model: {model_name}.")


def get_residual_stream_activations(model,
                                    tokenizer,
                                    prompt: str,
                                    n_samples: int = 1
):
    """ Returns a torch.Tensor of activations. 

    activations.shape = torch.Size([batch, seq, num_hidden_layers, hidden_size])
    """

    #model, tokenizer = models.get_model_from_name(model_name)
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True)

    activations = torch.zeros((input_ids.shape[0], input_ids.shape[1], model.config.num_hidden_layers, model.config.hidden_size)) # batch, seq, n_layers, d_model

    def get_activation(layer_id):
        def hook(model, input, output):
            activations[:, :, layer_id, :] = output[0].detach()
        return hook

    layers_to_enum = get_layers_to_enumerate(model)
    hooks = []
    for i, layer in enumerate(layers_to_enum):
        hook_handle = layer.register_forward_hook(get_activation(i))
        hooks.append(hook_handle)

    _ = model(input_ids)
    for h in hooks:
        h.remove()
    return activations


def apply_activation_patch(model, 
                           tokenizer, 
                           target_prompt,
                           source_activations, #: Float[torch.Tensor, 'batch seq n_layers d_model'], 
                           target_layer_idx: int = 2, 
                           target_token_idx: int = -1, 
):

    def patch_activations(source_activations, 
                        target_layer_idx, 
                        target_token_idx):
        def hook(module, input, output):
            output[0][:, target_token_idx, :] = source_activations[:, target_token_idx, target_layer_idx, :]
        return hook

    input_ids = tokenizer(target_prompt, return_tensors="pt", truncation=True)
    
    layers = get_layers_to_enumerate(model)
    target_layer = layers[target_layer_idx]
    hook_handle = target_layer.register_forward_hook(
        patch_activations(source_activations, target_layer_idx, target_token_idx)
    )
    
    try:
        with torch.no_grad():
            tokens = model.generate(**input_ids)
    finally:
        hook_handle.remove()
    
    return tokens



def request_patching(model_name: str,
                     source_prompt: str,
                     target_prompt: str,
                     target_layer_idx: int):

    model, tokenizer = get_model_from_name(model_name)
    layers = len(get_layers_to_enumerate(model))

    activations = get_residual_stream_activations(model,
                                                tokenizer,
                                                source_prompt)


    for layer in range(layers):
        tokens = apply_activation_patch(model=model,
                                        tokenizer=tokenizer,
                                        target_prompt=target_prompt,
                                        target_layer_idx=target_layer_idx,
                                        source_activations=activations)

        str_tokens = tokenizer.batch_decode(tokens)
        last_str_token = str_tokens[-1].split()[-1] #corriger pour batch
        print(f'Layer {layer} - {last_str_token}')
    return