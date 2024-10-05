import torch
from tqdm import tqdm
import json
import pickle
import multiprocessing
import os
import importlib

from request_patching import request_patch_one_pair, create_patch_request_dict, baseline_completion, baseline_completion_plus
from models import get_model_from_name

import warnings
warnings.filterwarnings("ignore")


def run_patching(prompt_dict_filename: str,
                 model_name: str,
                 layers: list[int] = [14, 15, 16, 17, 18, 19, 20],
                 output_file_name: str = 'programming_problems.pkl',
                 output_folder: str = 'outputs'
):
    """
    
    Inputs
    ------
    prompt_dict_filename: str 
        Should be located in the 'data' folder. 
    """

    model, tokenizer = get_model_from_name(model_name)

    # 1. LOAD PROMPT DICT
    if os.path.splitext(prompt_dict_filename)[1].lower() == '.py':
        # if the prompt dict is in a python file
        module_name = prompt_dict_filename[:-3]  # Remove .py extension
        spec = importlib.util.spec_from_file_location(module_name, prompt_dict_filename)
        prompt_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompt_module)
        prompt_dict = prompt_module.prompt_dict
    else:
        # if the prompt dict is in a json file
        with open(os.path.join('data', prompt_dict_filename), 'r') as file:
            prompt_dict = json.load(file)
            

    # 2. BASELINE COMPLETION
    reverse_dict = {}
    for p in tqdm(prompt_dict, desc='Baseline completion'):
        context = prompt_dict[p]['context']
        prompt_dict[p]['baseline_completion'] = baseline_completion(context, model, tokenizer)
        reverse_dict[f"{prompt_dict[p]['R']}_{prompt_dict[p]['C']}"] = p

    # 3. PATCHING EXPERIMENTS
    patch_dict = {}
    pair_id = 0
    for ctx1 in tqdm(prompt_dict, desc='Context 1'):
        for ctx2 in tqdm(prompt_dict, desc='Context 2'):

            if ctx1 == ctx2:
                continue
            
            context_1 = prompt_dict[ctx1]['context']
            context_2 = prompt_dict[ctx2]['context']
            R1, R2 = prompt_dict[ctx1]['R'], prompt_dict[ctx2]['R']
            C1, C2 = prompt_dict[ctx1]['C'], prompt_dict[ctx2]['C']

            
            if f'{R1}_{C2}' not in reverse_dict or f'{R2}_{C1}' not in reverse_dict:
                continue 
                #R1C2_id, R2C1_id = reverse_dict[f'{R1}_{C1}'], reverse_dict[f'{R2}_{C2}']
            
            R1C2_id, R2C1_id = reverse_dict[f'{R1}_{C2}'], reverse_dict[f'{R2}_{C1}']

            token_per_layer, logits_per_layer = request_patch_one_pair(context_1, context_2, model, tokenizer, layers=layers)
            
            patch_dict[f'pair_{pair_id}'] = {'context_1': context_1,
                                             'context_2': context_2,
                                             'R1': R1,
                                             'R2': R2,
                                             'C1': C1,
                                             'C2': C2,
                                             'R1_C1': prompt_dict[ctx1]['baseline_completion'],
                                             'R2_C2': prompt_dict[ctx2]['baseline_completion'],
                                             'R1_C2': prompt_dict[R1C2_id]['baseline_completion'],
                                             'R2_C1': prompt_dict[R2C1_id]['baseline_completion'],
            }

            R1C1_index = tokenizer.convert_tokens_to_ids(prompt_dict[ctx1]['baseline_completion'])
            R2C1_index = tokenizer.convert_tokens_to_ids(prompt_dict[R2C1_id]['baseline_completion'])
            R1C2_index = tokenizer.convert_tokens_to_ids(prompt_dict[R1C2_id]['baseline_completion'])
            R2C2_index = tokenizer.convert_tokens_to_ids(prompt_dict[ctx2]['baseline_completion'])

            for j, layer_id in enumerate(layers):
                patch_dict[f'pair_{pair_id}'][f'token_l{layer_id}'] = token_per_layer[j]
                #top_index = torch.argmax(logits_per_layer[j], dim=-1)[0, -1].item()
                #top_logit = logits_per_layer[j][0, -1, top_index].item()
                #patch_dict[f'pair_{pair_id}'][f'logit_l{layer_id}'] = top_logit
                probabilities = torch.nn.functional.softmax(logits_per_layer[j], dim=-1)[0, -1].cpu().numpy()

                R1C1_proba = probabilities[R1C1_index]
                R2C1_proba = probabilities[R2C1_index]
                R1C2_proba = probabilities[R1C2_index]
                R2C2_proba = probabilities[R2C2_index]
                R1C1_logit = logits_per_layer[j][0, -1, R1C1_index].item()
                R2C1_logit = logits_per_layer[j][0, -1, R2C1_index].item()
                R1C2_logit = logits_per_layer[j][0, -1, R1C2_index].item()
                R2C2_logit = logits_per_layer[j][0, -1, R2C2_index].item()

                patch_dict[f'pair_{pair_id}'][f'R1C1_proba_l{layer_id}'] = R1C1_proba
                patch_dict[f'pair_{pair_id}'][f'R2C1_proba_l{layer_id}'] = R2C1_proba
                patch_dict[f'pair_{pair_id}'][f'R1C2_proba_l{layer_id}'] = R1C2_proba
                patch_dict[f'pair_{pair_id}'][f'R2C2_proba_l{layer_id}'] = R2C2_proba
                patch_dict[f'pair_{pair_id}'][f'R1C1_logit_l{layer_id}'] = R1C1_logit
                patch_dict[f'pair_{pair_id}'][f'R2C1_logit_l{layer_id}'] = R2C1_logit
                patch_dict[f'pair_{pair_id}'][f'R1C2_logit_l{layer_id}'] = R1C2_logit
                patch_dict[f'pair_{pair_id}'][f'R2C2_logit_l{layer_id}'] = R2C2_logit
                #patch_dict[f'pair_{pair_id}'][f'proba_l{layer_id}'] = probabilities[top_index]

            pair_id += 1


            # 4. SAVE PICKLE FILE
            with open(f'{output_folder}/{output_file_name}', 'wb') as f:
                pickle.dump(patch_dict, f)

    return prompt_dict




if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device =', device)
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")

    else:
        print("CUDA is not available. Listing CPUs instead.")
        print(multiprocessing.cpu_count())

    prompt_dict = run_patching(prompt_dict_filename="data/programming_problems_bis.py",
                               model_name="gemma-7b",
                               layers=[9, 10, 11, 12, 13, 14, 15, 16],
                               output_file_name='programming_problems_bis_gemma-7b.pkl')
