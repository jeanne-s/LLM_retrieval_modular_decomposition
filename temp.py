import torch
import torch.nn as nn
from torch import Tensor
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM
from jaxtyping import Float, Int
from typing import List, Optional, Tuple, Dict
import sys
from functools import partial
from tqdm import tqdm
import itertools
import json
import seaborn as sns
import pandas as pd
import multiprocessing
import pickle

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



prompt_dict = {
    'prompt_1': {'context': 'Emily is a dynamic Democrat who tirelessly works towards advancing civil rights, healthcare access, and public education reform. She combines her expertise in law and social work to advocate for policies that dismantle systemic inequalities and support mental health initiatives.',
                 'sex': 'f',
                 'party': 'd'
    },
    'prompt_2': {'context': "Carlos is an energetic Democrat with a focus on immigration reform and workers' rights. As the son of immigrants, he brings a personal passion to his advocacy, aiming to create pathways to citizenship and fair labor practices",
                 'sex': 'm',
                 'party': 'd'
    },
    'prompt_3': {'context': "Ethan is a steadfast Republican with a deep commitment to economic development and job creation. As an entrepreneur, he understands the challenges faced by small businesses and advocates for lower taxes and reduced government regulation to encourage investment and growth.",
                 'sex': 'm',
                 'party': 'r'
    },
    'prompt_4': {'context': "Rachel is a principled Republican who values national security and law enforcement. With a background in criminal justice, she advocates for policies that support the military and police forces, believing in the importance of a safe and secure society.",
                 'sex': 'f',
                 'party': 'r'
    },
    'prompt_5': {'context': "Anna is a passionate and dedicated Democrat, known for her strong commitment to social justice, healthcare reform, and environmental protection. With a background in public policy, she actively participates in community organizing and voter registration drives, aiming to empower underrepresented groups and promote policies that ensure equality and sustainability.",
                 'sex': 'f',
                 'party': 'd'
    },
    'prompt_6': {'context': "Bob is a devoted Republican, characterized by his firm belief in limited government, fiscal responsibility, and strong national defense. With a background in business, he champions policies that promote economic growth, entrepreneurship, and tax reform, advocating for a regulatory environment that fosters innovation and competition.",
                 'sex': 'f',
                 'party': 'r'}
}

few_shot_ex = 'Sophia is a dedicated Democrat with a strong focus on social justice and environmental sustainability. As a community organizer, she recognizes the struggles of underserved communities and champions policies for equitable access to healthcare and education. In 2020 she voted for Mr. Biden.'


def patch_no_dialog_few_shot(prompt_dict: dict,
                             few_shot_paragraph: str,
                             years: list[str] = ['1992', '1996', '2000', '2004', '2008', '2012'],
                             layers: list[int] = [14, 15, 16, 17, 18, 19, 20],
                             output_file_name: str = 'no_dialog_patching_dict.pkl',
                             model_name: str = 'pythia-6.9b'
):

    model, tokenizer = get_model_from_name(model_name)

    # BASELINE COMPLETION
    for p in tqdm(prompt_dict, desc='Baseline completion'):
        pronoun = 'he' if prompt_dict[p]['sex']=='m' else 'she'
        prompt_dict[p]['baseline_completion'] = {}
        for year in years:
            context = f"{few_shot_paragraph}\n {prompt_dict[p]['context']} In {year} {pronoun} voted for Mr."
            prompt_dict[p]['baseline_completion'][f'{year}'] = baseline_completion(context, model, tokenizer)


    patch_dict = {}
    pair_id = 0

    for char_1 in tqdm(prompt_dict, desc='char_1'):
        for year_1 in years:
            for char_2 in prompt_dict:
                for year_2 in years:

                    if (char_1 == char_2 and 
                        year_1 == year_2):
                        # in this case context_1 = context_2
                        continue

                    party_1 = prompt_dict[char_1]['party']
                    party_2 = prompt_dict[char_2]['party']
                    if (party_1 == party_2):
                        continue

                    pronouns = ['he' if prompt_dict[x]['sex']=='m' else 'she' for x in [char_1, char_2]]
                    context_1 = f"{few_shot_paragraph}\n {prompt_dict[char_1]['context']} In {year_1} {pronouns[0]} voted for Mr."
                    context_2 = f"{few_shot_paragraph}\n {prompt_dict[char_2]['context']} In {year_2} {pronouns[1]} voted for Mr."
                    
                    token_per_layer, logits_per_layer = request_patch_one_pair(context_1, context_2, model, tokenizer, layers=layers)
                    patch_dict[f'pair_{pair_id}'] = {'context_1': context_1,
                                                    'context_2': context_2,
                                                    'year_1': year_1,
                                                    'year_2': year_2,
                                                    'R1_C1': prompt_dict[char_1]['baseline_completion'][f'{year_1}'],
                                                    'R2_C2': prompt_dict[char_2]['baseline_completion'][f'{year_2}'],
                                                    'R1_C2': prompt_dict[char_1]['baseline_completion'][f'{year_2}'],
                                                    'R2_C1': prompt_dict[char_2]['baseline_completion'][f'{year_1}']
                    }
                    for j, layer_id in enumerate(layers):
                        patch_dict[f'pair_{pair_id}'][f'token_l{layer_id}'] = token_per_layer[j]

                        probabilities = torch.nn.functional.softmax(logits_per_layer[j], dim=-1)[0, -1].cpu().numpy() 
                        patch_dict[f'pair_{pair_id}'][f'logit_l{layer_id}'] = logits_per_layer[j]
                    pair_id += 1

                    with open(f'outputs/{output_file_name}', 'wb') as f:
                        pickle.dump(patch_dict, f)
    
    print(f'Number of patchings: {pair_id}')
    return 


patch_no_dialog_few_shot(prompt_dict=prompt_dict,
                         few_shot_paragraph=few_shot_ex,
                         output_file_name='no_dialog_patching_dict_fewshot.pkl'
)