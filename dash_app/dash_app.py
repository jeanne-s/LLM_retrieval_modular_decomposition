from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd
import json
import pickle
import plotly.express as px

# TODO: automatic set of pair-dropdown instead of hardcoded 



app = Dash(__name__)

app.layout = html.Div([
    html.Div(className='row', children=[
        dcc.Dropdown(['gpt2-small', 'pythia-1b', 'pythia-2.8b'], value='gpt2-small', id='model-dropdown')
    ]),
    html.Div(className='row', children=[
        dcc.Dropdown(['short_stories', 'dialogs'], value='short_stories', id='task-dropdown')
    ]),
    html.Div(className='row', children=[
        dcc.Dropdown(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], value='0', id='pair-dropdown'),
        html.Div([
        html.B('Context 1:'),
        html.P(id='pair-context1-container'),
        html.P(id='context1-baseline-completion')
        ]),
        html.Div([
        html.B('Context 2:'),
        html.P(id='pair-context2-container'),
        html.P(id='context2-baseline-completion')
        ])
    ]),

    html.Div(className='row', children=[
        html.B('Output tokens after patching each layer:'),
        html.P(id='out-tokens-container')
    ]),

    dcc.Graph(figure={}, id='accuracy-graph-one-pair'),
    dcc.Graph(figure={}, id='accuracy-graph-all-pairs')
])


### CONTEXT 1 ###
@callback(
    Output('pair-context1-container', 'children'),
    [Input('pair-dropdown', 'value'),
     Input('model-dropdown', 'value'),
     Input('task-dropdown', 'value')]
)
def update_output(pair, model_name, task):
    with open(f'patch_request_dictionaries/{model_name}_{task}.pkl', 'rb') as f:
        patch_request_dict = pickle.load(f)
    return patch_request_dict[f'pair_{pair}']['context_1']


### CONTEXT 2 ###
@callback(
    Output('pair-context2-container', 'children'),
    [Input('pair-dropdown', 'value'),
     Input('model-dropdown', 'value'),
     Input('task-dropdown', 'value')]
)
def update_output(pair, model_name, task):
    with open(f'patch_request_dictionaries/{model_name}_{task}.pkl', 'rb') as f:
        patch_request_dict = pickle.load(f)
    return patch_request_dict[f'pair_{pair}']['context_2']


### CONTEXT 1 BASELINE COMPLETION ###
@callback(
    Output('context1-baseline-completion', 'children'),
    [Input('pair-dropdown', 'value'),
     Input('model-dropdown', 'value'),
     Input('task-dropdown', 'value')]
)
def update_output(pair, model_name, task):
    with open(f'patch_request_dictionaries/{model_name}_{task}.pkl', 'rb') as f:
        patch_request_dict = pickle.load(f)

    if task == 'short_stories':
        return f"Baseline completion : {patch_request_dict[f'pair_{pair}']['R1_C1']}"
    elif task == 'dialogs':
        return f"Baseline completion : {patch_request_dict[f'pair_{pair}']['R_C1']}"


### CONTEXT 2 BASELINE COMPLETION ###
@callback(
    Output('context2-baseline-completion', 'children'),
    [Input('pair-dropdown', 'value'),
     Input('model-dropdown', 'value'),
     Input('task-dropdown', 'value')]
)
def update_output(pair, model_name, task):
    with open(f'patch_request_dictionaries/{model_name}_{task}.pkl', 'rb') as f:
        patch_request_dict = pickle.load(f)

    if task == 'short_stories':
        return f"Baseline completion : {patch_request_dict[f'pair_{pair}']['R2_C2']}"
    elif task == 'dialogs':
        return f"Baseline completion : {patch_request_dict[f'pair_{pair}']['R_C2']}"


### OUTPUT TOKENS PER LAYER ###
@callback(
    Output('out-tokens-container', 'children'),
    [Input('pair-dropdown', 'value'),
     Input('model-dropdown', 'value'),
     Input('task-dropdown', 'value')]
)
def update_output(pair, model_name, task):
    out_tokens = []
    with open(f'patch_request_dictionaries/{model_name}_{task}.pkl', 'rb') as f:
        patch_request_dict = pickle.load(f)
    
    patching_result = patch_request_dict[f'pair_{pair}']['patching_result']
    for l, token in enumerate(patching_result):
        out_tokens.append(f'{l}:{token},  ')
    return out_tokens


### ACCURACY GRAPH ONE PAIR ###
@callback(
    Output(component_id='accuracy-graph-one-pair', component_property='figure'),
    [Input('pair-dropdown', 'value'),
     Input('model-dropdown', 'value'),
     Input('task-dropdown', 'value')]
)
def update_graph_one_pair(pair, model_name, task):
    with open(f'patch_request_dictionaries/{model_name}_{task}.pkl', 'rb') as f:
        patch_request_dict = pickle.load(f)

    accuracy_df = pd.DataFrame(columns=['request_context', 'layer', 'acc'])

    if task == 'short_stories':
        R2_C2 = patch_request_dict[f'pair_{pair}']['R2_C2']
        R1_C1 = patch_request_dict[f'pair_{pair}']['R1_C1']
        R1_C2 = patch_request_dict[f'pair_{pair}']['R1_C2']
        for l, layer_output in enumerate(patch_request_dict[f'pair_{pair}']['patching_result']):
            for request_context, name in zip([R2_C2, R1_C1, R1_C2], [f'R2(C2): {R2_C2}', f'R1(C1): {R1_C1}', f'R1(C2): {R1_C2}']):
                acc = 1 if request_context == layer_output else 0
                temp_series = pd.DataFrame([{'request_context': name, 'layer': l, 'acc': acc}])
                accuracy_df = pd.concat([accuracy_df, temp_series], ignore_index=True)
    elif task == 'dialogs':
        # TODO write sub-function for dialogs
        return

    fig = px.line(accuracy_df, x='layer', y='acc', color='request_context', 
                  title='accuracy',
                  width=800,
                  height=500)
    return fig


### ACCURACY GRAPH ALL PAIRS ###
@callback(
    Output(component_id='accuracy-graph-all-pairs', component_property='figure'),
    [Input('model-dropdown', 'value'),
     Input('task-dropdown', 'value')]
)
def update_graph_all_pairs(model_name, task):

    with open(f'patch_request_dictionaries/{model_name}_{task}.pkl', 'rb') as f:
        patch_request_dict = pickle.load(f)

    accuracy_df = pd.DataFrame(columns=['pair', 'request_context', 'layer', 'acc'])
    for i, pair in enumerate(patch_request_dict):
        R2_C2 = patch_request_dict[f'pair_{i}']['R2_C2']
        R1_C1 = patch_request_dict[f'pair_{i}']['R1_C1']
        R1_C2 = patch_request_dict[f'pair_{i}']['R1_C2']
        for l, layer_output in enumerate(patch_request_dict[f'pair_{i}']['patching_result']):
            for request_context, name in zip([R2_C2, R1_C1, R1_C2], ['R2(C2)', 'R1(C1)', 'R1(C2)']):
                acc = 1 if request_context == layer_output else 0
                temp_series = pd.DataFrame([{'pair': i, 'request_context': name, 'layer': l, 'acc': acc}])
                accuracy_df = pd.concat([accuracy_df, temp_series], ignore_index=True)

    agg_df = accuracy_df.groupby(['request_context', 'layer'], as_index=False)['acc'].mean()
    fig = px.line(agg_df, x='layer', y='acc',
                  color='request_context',
                  title='Accuracy on all pairs of prompts',
                  width=800,
                  height=500)
    return fig


if __name__ == '__main__':
    app.run(debug=True)
