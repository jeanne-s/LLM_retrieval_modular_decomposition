import json

from models import get_model_from_name
from activations import get_residual_stream_activations, apply_activation_patch, get_layers_to_enumerate


def create_all_prompt_pairs(stories_filepath: str = 'data/short_stories.json',
                            requests_filepath: str = 'data/requests.json'
):
    """ Creates a list of prompt pairs (context_1, context_2) for request patching 
    experiments.

    Returns
    -------
    all_prompt_pairs: list of prompt pairs in the form [(context_1, context_2), (context_1, context_2), ...]
    R2_C2: list of same length as all_prompt_pairs. Contains the result of request 2 applied on context_2
    R1_C2: list of same length as all_prompt_pairs. Contains the result of request 1 applied on context_2
    R1_C1: list of same length as all_prompt_pairs. Contains the result of request 1 applied on context_1
    """
    # TODO: On peut créer plus de prompts en inversant les requêtes

    stories = json.load(open(stories_filepath))
    requests = json.load(open(requests_filepath))
    print(f'Number of stories: {len(stories)}')
    print(f'Number of requests: {len(requests)}')

    all_prompt_pairs = []
    R2_C2 = []
    R1_C2 = []
    R1_C1 = []

    for i in range(0, len(stories)):
        for j in range(i+1, len(stories)):
            context_1 = stories[f'story_{i}']['context'] + '\n\n' + requests['request_0']['context']
            context_2 = stories[f'story_{j}']['context'] + '\n\n' + requests['request_1']['context']
            all_prompt_pairs.append([context_1, context_2])
            R1, R2 = requests['request_0']['type'], requests['request_1']['type']
            R2_C2.append(stories[f'story_{j}'][f'{R2}'])
            R1_C2.append(stories[f'story_{j}'][f'{R1}'])
            R1_C1.append(stories[f'story_{i}'][f'{R1}'])

    return all_prompt_pairs, R2_C2, R1_C2, R1_C1


def request_patch_all_prompt_pairs(model_name: str,
                                   all_prompt_pairs=None,
                                   details=False
):    
    """ Applies the request patching experiment (layer-wise) on all prompt 
    pairs obtained with the function create_all_promp_pairs.  
    
    Returns
    -------
    tokens_per_prompt_pair: list of list of same length as all_prompt_pairs. 
        Each element is a list of size model.n_layers, containing the output
        of the experiment when each layer is patched. 
    """
    model, tokenizer = get_model_from_name('gpt2-small')
    layers = len(get_layers_to_enumerate(model))
    tokens_per_prompt_pair = []

    if all_prompt_pairs==None:
        all_prompt_pairs, R2_C2, R1_C2, R1_C1 = create_all_prompt_pairs()

    for context_1, context_2 in all_prompt_pairs:

        if details:
            print('context_1', context_1)
            print('context_2', context_2)

        activations = get_residual_stream_activations(model,
                                                      tokenizer,
                                                      context_1)

        token_per_layer = []
        for layer in range(layers):
            tokens = apply_activation_patch(model=model,
                                            tokenizer=tokenizer,
                                            target_prompt=context_2,
                                            target_layer_idx=layer,
                                            source_activations=activations)

            str_tokens = tokenizer.batch_decode(tokens)
            last_str_token = str_tokens[-1].split()[-1]
            token_per_layer.append(last_str_token)

        if details:
            print(token_per_layer) 
        tokens_per_prompt_pair.append(token_per_layer)

    return tokens_per_prompt_pair, R2_C2, R1_C2, R1_C1