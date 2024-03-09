from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd
import json

app = Dash(__name__)

app.layout = html.Div([
    html.Div(className='row', children=[
        dcc.Dropdown(['gpt2-small', 'pythia-2.8b'], 'gpt2-small', id='model-dropdown'),
        html.Div(id='model-container')
    ]),
    html.Div(className='row', children=[
        dcc.Dropdown(['0', '1', '2', '3', '4'], '0', id='context1-dropdown'),
        html.Div(id='context1-container')
    ])
])


@callback(
    Output('model-container', 'children'),
    Input('model-dropdown', 'value')
)
def update_output(value):
    return f'You have selected {value}'

@callback(
    Output('context1-container', 'children'),
    Input('context1-dropdown', 'value')
)
def update_output(value):
    stories = json.load(open('../data/short_stories.json'))
    requests = json.load(open('../data/requests.json'))
    context_1 = stories[f'story_{value}']['context'] + '\n\n' + requests['request_0']['context']
    return context_1

if __name__ == '__main__':
    app.run(debug=True)
