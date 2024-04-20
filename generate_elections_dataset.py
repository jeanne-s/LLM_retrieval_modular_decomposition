import torch
from torch import Tensor
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM
from jaxtyping import Float, Int
import itertools
from functools import partial
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import multiprocessing

from request_patching import request_patch_one_pair, create_patch_request_dict, baseline_completion, baseline_completion_plus
from models import get_model_from_name

import warnings
warnings.filterwarnings("ignore")

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



def us_election_prompts(year: str):
    context_1 = f"""Alice: Have you been following the election coverage? I can't believe it's already November {year}. This election feels more significant than any I can remember. 
Bob: Yes, I've been keeping a close eye on it. This year, as a committed Republican, I find the election particularly pivotal. The Republican agenda, focusing on economic stability and national security, really resonates with me.
Alice: I'm siding with the Democrats myself. Their emphasis on healthcare and education matches what I believe our country desperately needs right now.
Bob: I understand where you're coming from, Alice. However, from my standpoint, the Republican principles of limited government and individual freedoms are what we need to navigate these challenging times.
Alice: I know exactly who I will vote for. I support Mr."""

    context_2 = f"""Alice: Have you been following the election coverage? I can't believe it's already November {year}. This election feels more significant than any I can remember. 
Bob: Yes, I've been keeping a close eye on it. This year, as a committed Republican, I find the election particularly pivotal. The Republican agenda, focusing on economic stability and national security, really resonates with me.
Alice: I'm siding with the Democrats myself. Their emphasis on healthcare and education matches what I believe our country desperately needs right now.
Bob: I understand where you're coming from, Alice. However, from my standpoint, the Republican principles of limited government and individual freedoms are what we need to navigate these challenging times.
Alice: I see your point, Bob. Regardless, {year} feels like a watershed moment.
Bob: I know exactly who I will vote for. I support Mr."""

    return context_1, context_2



model, tokenizer = get_model_from_name("pythia-6.9b")

years = ['1984', '1988', '1992', '1996', '2000', '2004', '2008', '2012']
characters = ['Alice', 'Bob']
layers = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30]
output_file_name = 'elections_patching_results.csv'

patching_df = pd.DataFrame(columns=['context_1', 
                                    'context_2',
                                    'character_1',
                                    'character_2',
                                    'year_1',
                                    'year_2',
                                    'R1_C1',
                                    'R1_C2',
                                    'R2_C2',
                                    'R2_C1'])


# PATCHING AT CONSTANT YEAR
for year in years:
    context_1A, context_1B = us_election_prompts(year)

    for i, (first, second) in enumerate([(context_1A, context_1B), (context_1B, context_1A)]):
        baseline_first =  baseline_completion(first, model, tokenizer)
        baseline_second =  baseline_completion(second, model, tokenizer)
        character_1 = 'Alice' if i==0 else 'Bob'
        character_2 = 'Bob' if i==0 else 'Alice'

        answer_list = request_patch_one_pair(baseline_first, baseline_second, model, tokenizer, layers=layers)
        temp_dict = {'context_1': baseline_first,
                    'context_2': baseline_second,
                    'character_1': character_1,
                    'character_2': character_2,
                    'year_1': year,
                    'year_2': year,
                    'R1_C1': baseline_first,
                    'R2_C2': baseline_second,
                    'R1_C2': baseline_first,
                    'R2_C1': baseline_second
        }
            
        for j, layer_id in enumerate(layers):
            temp_dict[f'patch_l{layer_id}'] = answer_list[j]
        
        df_dictionary = pd.DataFrame([temp_dict])
        patching_df = pd.concat([patching_df, df_dictionary], ignore_index=True)
        patching_df.to_csv(f'outputs/{output_file_name}', index=False)


# PATCHING AT CONSTANT CHARACTER
for character in characters:
    for pair in list(itertools.combinations(years, 2)):
        context_1A, context_1B = us_election_prompts(pair[0])
        context_2A, context_2B = us_election_prompts(pair[1])
        baseline_1A =  baseline_completion(context_1A, model, tokenizer)
        baseline_1B =  baseline_completion(context_1B, model, tokenizer)
        baseline_2A =  baseline_completion(context_2A, model, tokenizer)
        baseline_2B =  baseline_completion(context_2B, model, tokenizer)

        valid_patchings = [(context_1A, context_2A, pair[0], pair[1], baseline_1A, baseline_2A, baseline_1A, baseline_2A),
                           (context_1B, context_2B, pair[0], pair[1], baseline_1B, baseline_2B, baseline_1B, baseline_2B),
                           (context_2A, context_1A, pair[1], pair[0], baseline_2A, baseline_1A, baseline_2A, baseline_1A),
                           (context_2B, context_1B, pair[1], pair[0], baseline_2B, baseline_1B, baseline_2B, baseline_1B),
        ]

        for ctx_pair in tqdm(valid_patchings):
            character = "Alice" if ctx_pair[0].split(':')[-2][-1]=='e' else "Bob"

            answer_list = request_patch_one_pair(ctx_pair[0], ctx_pair[1], model, tokenizer, layers=layers)
            temp_dict = {'context_1': ctx_pair[0],
                        'context_2': ctx_pair[1],
                        'character_1': character,
                        'character_2': character,
                        'year_1': ctx_pair[2],
                        'year_2': ctx_pair[3],
                        'R1_C1': ctx_pair[4],
                        'R2_C2': ctx_pair[5],
                        'R1_C2': ctx_pair[6],
                        'R2_C1': ctx_pair[7]
            }

            for i, layer_id in enumerate(layers):
                temp_dict[f'patch_l{layer_id}'] = answer_list[i]
            
            df_dictionary = pd.DataFrame([temp_dict])
            patching_df = pd.concat([patching_df, df_dictionary], ignore_index=True)
            patching_df.to_csv(f'outputs/{output_file_name}', index=False)


# 2 DIM VARIATION
for pair in list(itertools.combinations(years, 2)):
    context_1A, context_1B = us_election_prompts(pair[0])
    context_2A, context_2B = us_election_prompts(pair[1])
    baseline_1A =  baseline_completion(context_1A, model, tokenizer)
    baseline_1B =  baseline_completion(context_1B, model, tokenizer)
    baseline_2A =  baseline_completion(context_2A, model, tokenizer)
    baseline_2B =  baseline_completion(context_2B, model, tokenizer)

    valid_patchings = [(context_1A, context_2B, pair[0], pair[1], baseline_1A, baseline_2B, baseline_2A, baseline_1B),
                       (context_1B, context_2A, pair[0], pair[1], baseline_1B, baseline_2A, baseline_2B, baseline_1A),
                       (context_2A, context_1B, pair[1], pair[0], baseline_2A, baseline_1B, baseline_1A, baseline_2B),
                       (context_2B, context_1A, pair[1], pair[0], baseline_2B, baseline_1A, baseline_1B, baseline_2A),
    ]
    

    for ctx_pair in tqdm(valid_patchings):

        character_1 = "Alice" if ctx_pair[0].split(':')[-2][-1]=='e' else "Bob"
        character_2 = "Alice" if character_1=="Bob" else "Bob"
        assert character_1 != character_2

        answer_list = request_patch_one_pair(ctx_pair[0], ctx_pair[1], model, tokenizer, layers=layers)
        temp_dict = {'context_1': ctx_pair[0],
                    'context_2': ctx_pair[1],
                    'character_1': character_1,
                    'character_2': character_2,
                    'year_1': ctx_pair[2],
                    'year_2': ctx_pair[3],
                    'R1_C1': ctx_pair[4],
                    'R2_C2': ctx_pair[5],
                    'R1_C2': ctx_pair[6],
                    'R2_C1': ctx_pair[7]
        }
        
        # Patching
        for i, layer_id in enumerate(layers):
            temp_dict[f'patch_l{layer_id}'] = answer_list[i]
        
        df_dictionary = pd.DataFrame([temp_dict])
        patching_df = pd.concat([patching_df, df_dictionary], ignore_index=True)
        patching_df.to_csv(f'outputs/{output_file_name}.csv', index=False)

    

