import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

from request_patching import request_patch_all_prompt_pairs

def plot_request_patching_accuracy(model_name: str,
                                   savefig = False,
                                   details = False
):
    """ Plots the normalized accuracy from the request_patching experiment.
    """
    accuracy_df = pd.DataFrame(columns=['pair', 'request_context', 'layer', 'acc'])
    tokens_per_prompt_pair, R2_C2, R1_C2, R1_C1 = request_patch_all_prompt_pairs(model_name=model_name,
                                                                                 details=details)

    for i, pair in enumerate(tokens_per_prompt_pair): 
        for l, layer_output in enumerate(pair):
            for request_context, name in zip([R2_C2, R1_C1, R1_C2], ['R2(C2)', 'R1(C1)', 'R1(C2)']):
                acc = 1 if request_context[i] == layer_output else 0 # previously == -> in
                temp_series = pd.DataFrame([{'pair': i, 'request_context': name, 'layer': l, 'acc': acc}])
                accuracy_df = pd.concat([accuracy_df, temp_series], ignore_index=True)    

    fig = sns.lineplot(data=accuracy_df, x='layer', y='acc', hue='request_context').set(title=f'{model_name}')
    if savefig:
        plt.savefig(f"figures/request_patching_normalized_accuracy_{model_name}.png")
        
    return fig  