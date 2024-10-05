import json
import pickle
import random
import torch

from models import get_model_from_name
from activations import get_residual_stream_activations, apply_activation_patch, get_layers_to_enumerate



def create_patch_request_dict(model_name: str,
                              dataset: str = 'short_stories',
                              details=False):

    assert dataset in ['short_stories', 'dialogs', 'dialogs_2']
    model, tokenizer = get_model_from_name(model_name)

    if dataset == 'short_stories':
        _,_,_,_,dict = create_all_prompt_pairs_short_stories(tokenizer)
    elif dataset == 'dialogs':
        _,_,_,dict = create_all_prompt_pairs_dialogs(tokenizer, model)
    elif dataset == 'dialogs_2':
        dict = create_prompt_pairs_dialogs_2(model, tokenizer)
    
    
    for pair in dict:
        dict[pair]['patching_result'] = request_patch_one_pair(context_1=dict[pair]['context_1'],
                                                               context_2=dict[pair]['context_2'],
                                                               model=model,
                                                               tokenizer=tokenizer,
                                                               details=details)[0]
        

    with open(f'dash_app/patch_request_dictionaries/{model_name}_{dataset}.pkl', 'wb') as f:
        pickle.dump(dict, f)
    return dict


def create_all_prompt_pairs_short_stories(tokenizer,
                                          stories_filepath: str = 'data/short_stories.json',
                                          requests_filepath: str = 'data/requests.json',
):
    """ Creates a list of prompt pairs (context_1, context_2) for request patching experiments.

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
    prompt_pairs_dict = {}

    pair_id = 0
    for i in range(0, len(stories)):
        for j in range(i+1, len(stories)):
            context_1 = stories[f'story_{i}']['context'] + '\n\n' + requests['request_0']['context']
            context_2 = stories[f'story_{j}']['context'] + '\n\n' + requests['request_1']['context']
            all_prompt_pairs.append([context_1, context_2])
            R1, R2 = requests['request_0']['type'], requests['request_1']['type']
            R2_C2.append(stories[f'story_{j}'][f'{R2}'])
            R1_C2.append(stories[f'story_{j}'][f'{R1}'])
            R1_C1.append(stories[f'story_{i}'][f'{R1}'])

            # Dict
            prompt_pairs_dict[f'pair_{pair_id}'] = {}
            prompt_pairs_dict[f'pair_{pair_id}']['context_1'] = context_1
            prompt_pairs_dict[f'pair_{pair_id}']['context_2'] = context_2
            prompt_pairs_dict[f'pair_{pair_id}']['R1_C1'] = get_first_token_from_str(stories[f'story_{i}'][f'{R1}'],
                                                                                     tokenizer)
            prompt_pairs_dict[f'pair_{pair_id}']['R1_C2'] = get_first_token_from_str(stories[f'story_{j}'][f'{R1}'],
                                                                                     tokenizer)
            prompt_pairs_dict[f'pair_{pair_id}']['R2_C2'] = get_first_token_from_str(stories[f'story_{j}'][f'{R2}'],
                                                                                     tokenizer)

            pair_id += 1

    return all_prompt_pairs, R2_C2, R1_C2, R1_C1, prompt_pairs_dict


def create_all_prompt_pairs_dialogs(tokenizer,
                                    model,
                                    dialog_filepath: str = 'data/dialogs.json'):

    # TODO: faire tourner la pipeline dans les deux sens !
    dialogs = json.load(open(dialog_filepath))
    print(f'Number of stories: {len(dialogs)}')

    all_prompt_pairs = []
    R_C2 = []
    R_C1 = []
    prompt_pairs_dict = {}

    pair_id = 0
    for i in range(0, len(dialogs)):
        for j in range(0, len(dialogs)):
            if i !=j:
                context_1 = dialogs[f'dialog_{i}']['context']
                context_2 = dialogs[f'dialog_{j}']['context']
                all_prompt_pairs.append([context_1, context_2])
                R_C2.append(dialogs[f'dialog_{j}']['emotion'])
                R_C1.append(dialogs[f'dialog_{i}']['emotion'])

                # Dict
                prompt_pairs_dict[f'pair_{pair_id}'] = {}
                prompt_pairs_dict[f'pair_{pair_id}']['context_1'] = context_1
                prompt_pairs_dict[f'pair_{pair_id}']['context_2'] = context_2

                prompt_pairs_dict[f'pair_{pair_id}']['R_C1'] = baseline_completion(context=context_1,
                                                                                   model=model,
                                                                                   tokenizer=tokenizer)
                prompt_pairs_dict[f'pair_{pair_id}']['R_C2'] = baseline_completion(context=context_2,
                                                                                   model=model,
                                                                                   tokenizer=tokenizer)
                pair_id += 1

    return all_prompt_pairs, R_C1, R_C2, prompt_pairs_dict






def request_patch_all_prompt_pairs(model_name: str,
                                   dataset: str = 'short_stories',
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
    assert dataset in ['short_stories', 'dialogs']
    model, tokenizer = get_model_from_name(model_name)
    tokens_per_prompt_pair = []

    if dataset == 'short_stories':
        all_prompt_pairs, R2_C2, R1_C2, R1_C1, prompt_pairs_dict = create_all_prompt_pairs_short_stories(tokenizer)
        R2_C2 = get_first_token_from_str_list(R2_C2, tokenizer)
        R1_C2 = get_first_token_from_str_list(R1_C2, tokenizer)
        R1_C1 = get_first_token_from_str_list(R1_C1, tokenizer)
    elif dataset == 'dialogs':
        all_prompt_pairs, R_C1, R_C2 = create_all_prompt_pairs_dialogs()
        R_C1 = get_first_token_from_str_list(R_C1, tokenizer)
        R_C2 = get_first_token_from_str_list(R_C2, tokenizer)

    for context_1, context_2 in all_prompt_pairs:

        token_per_layer, logits_per_layer = request_patch_one_pair(context_1=context_1,
                                                    context_2=context_2,
                                                    model=model,
                                                    tokenizer=tokenizer,
                                                    details=details)
        
        tokens_per_prompt_pair.append(token_per_layer)
    
    if dataset == 'short_stories':
        return tokens_per_prompt_pair, R2_C2, R1_C2, R1_C1
    elif dataset == 'dialogs':
        return tokens_per_prompt_pair, R_C1, R_C2
    


def request_patch_one_pair(context_1: str,
                           context_2: str,
                           model,
                           tokenizer,
                           layers=None,
                           top_k_logits=1,
                           details=False
):
    """ Applies request patching layer-wise from context_1 to context_2.
    """
    
    if details:
        print(f'context_1: {context_1}')
        print(f'context_2: {context_2}')

    activations = get_residual_stream_activations(model,
                                                  tokenizer,
                                                  context_1)

    token_per_layer = []
    logits_per_layer = []
    if layers == None:
        layers = len(get_layers_to_enumerate(model))
        layers = [i for i in range(0, layers)]

    for layer in layers:
        tokens, original_length, logits = apply_activation_patch(model=model,
                                                                 tokenizer=tokenizer,
                                                                 target_prompt=context_2,
                                                                 target_layer_idx=layer,
                                                                 source_activations=activations)

        last_token = tokens[0, original_length-1] # batch 0, last token
        last_str_token = tokenizer.decode(last_token)
        token_per_layer.append(last_str_token)
        logits_per_layer.append(logits)

    if details:
        print(token_per_layer)
    return token_per_layer, logits_per_layer


def get_first_token_from_str(string: str,
                             tokenizer,
                             task: str = 'short_stories'
) -> str:
    """ Returns the first token from a given word on the prompt: 'The answer is {string}'.
    This function is useful to get the accuracy plots (plots/plot_request_patching_accuracy)

    Examples:
        - if tokenization('The answer is Daniel') = 'The', 'answer', 'is', ' Dan', 'iel'
          get_first_token_from_str('Daniel') = ' Dan' (notice the space)
        - if tokenization('The answer is Daniel') = 'The', 'answer', 'is', 'Daniel'
          get_first_token_from_str('Daniel') = 'Daniel'
    """
    # TODO bizarre cette fonction
    assert task in ['short_stories', 'dialogs', 'dialogs_2']
    if task in ['short_stories', 'dialogs']:
        tokenized_ids = tokenizer.encode(f'The answer is {string}')
    elif task == 'dialogs_2':
        tokenized_ids = tokenizer.encode(f'{string}')

    for token_id in reversed(tokenized_ids):
        str_token = tokenizer.decode(token_id)
        if string[0] in str_token:
            return str_token 


def get_first_token_from_str_list(string_list,
                                  tokenizer
):
    """ Applies get_first_token_from_str successively to all elements of the
    input string_list.
    """
    first_tokens_list = []
    for string in string_list:
        first_tokens_list.append(get_first_token_from_str(string, tokenizer))

    return first_tokens_list


def baseline_completion(context: str,
                        model,
                        tokenizer
) -> str:
    """ Returns the str token predicted by the given model on the given prompt.
    """
    input_ids = tokenizer(context, return_tensors="pt", truncation=True)
    outputs = model(**input_ids)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    last_token = torch.argmax(probabilities, dim=-1)[0, -1]
    last_str_token = tokenizer.decode(last_token)
    """
    tokens = model.generate(**input_ids, 
                            pad_token_id=tokenizer.eos_token_id,
                            max_new_tokens=10)
    return tokenizer.decode(tokens[0, -1])
    """
    return last_str_token


def baseline_completion_plus(context: str, 
                             model,
                             tokenizer,
                             max_new_tokens: int = 5,
):
    # TODO completion sur plusieurs tokens
    input_ids = tokenizer(context, return_tensors="pt", truncation=True).input_ids

    if hasattr(model.config, "max_new_tokens"):
        tokens = model.generate(input_ids,
                                max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.eos_token_id)
    else:
        max_length = input_ids.shape[1] + max_new_tokens
        tokens = model.generate(input_ids,
                                max_length=max_length,
                                pad_token_id=tokenizer.eos_token_id)
    
    generated_text = tokenizer.decode(tokens[0, -max_new_tokens:], skip_special_tokens=True)
    
    return generated_text



def create_prompt_pairs_dialogs_2(model,
                                  tokenizer,
                                  dialog_filepath: str = 'data/dialogs_2.json'):
    # TODO la fonction peut-être réécrite pour générer plus de pairs
    
    dialogs = json.load(open(dialog_filepath))
    print(f'Number of stories: {len(dialogs)}')

    prompt_pairs_dict = {}

    pair_id = 0
    for i in range(0, len(dialogs)):
        for j in range(0, len(dialogs)):
            if i!=j:
                prompt_pairs_dict[f'pair_{pair_id}'] = {}
                context_1 = dialogs[f'dialog_{i}']['context_1']
                context_2 = dialogs[f'dialog_{j}']['context_2']

                emotions = [dialogs[f'dialog_{i}'][f'attribute_character_2']]
                emotions.append(dialogs[f'dialog_{j}'][f'attribute_character_2'])
                emotions.append(dialogs[f'dialog_{j}'][f'attribute_character_1'])
                random.shuffle(emotions)

                request = f" If I had to choose between '{emotions[0]}', '{emotions[1]}', and '{emotions[2]}', I would say I felt '"
                #request_2 = f"{dialogs[f'dialog_{j}']['character_1']}: If I had to choose between '{emotions[0]}', '{emotions[1]}', '{emotions[2]}', and '{emotions[3]}', I would say I feel '"
                prompt_pairs_dict[f'pair_{pair_id}']['context_1'] = context_1 + request
                prompt_pairs_dict[f'pair_{pair_id}']['context_2'] = context_2 + request

                prompt_pairs_dict[f'pair_{pair_id}']['R1_C1'] = baseline_completion(context_1 + request, model, tokenizer)
                prompt_pairs_dict[f'pair_{pair_id}']['R1_C2'] = baseline_completion(dialogs[f'dialog_{j}']['context_1'] + request, model, tokenizer)
                prompt_pairs_dict[f'pair_{pair_id}']['R2_C2'] = baseline_completion(context_2 + request, model, tokenizer)

                """
                prompt_pairs_dict[f'pair_{pair_id}']['R1_C1'] = get_first_token_from_str(dialogs[f'dialog_{i}']['attribute_character_2'],
                                                                                         tokenizer,
                                                                                         task='dialogs_2')
                prompt_pairs_dict[f'pair_{pair_id}']['R1_C2'] = get_first_token_from_str(dialogs[f'dialog_{j}']['attribute_character_2'],
                                                                                         tokenizer,
                                                                                         task='dialogs_2')
                prompt_pairs_dict[f'pair_{pair_id}']['R2_C2'] = get_first_token_from_str(dialogs[f'dialog_{j}']['attribute_character_1'],
                                                                                         tokenizer,
                                                                                         task='dialogs_2')
                """
                pair_id += 1
    
    return prompt_pairs_dict