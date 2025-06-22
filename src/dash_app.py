from dash import dcc, Output, Input, callback, html, Dash
import pandas as pd
import plotly.express as px

df = pd.read_csv('../data/spacex_launch_dash.csv')
max_payload = df['Payload Mass (kg)'].max()
min_payload = df['Payload Mass (kg)'].min()

app = Dash(__name__)

app.layout = html.Div([
    html.H1('SpaceX Launch Records Dashboard',
            style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
    dcc.Dropdown(
        id='site-dropdown',
        options=[{'label': 'All Sites', 'value': 'ALL'},
                {'label': 'CCAFS LC-40', 'value': 'CCAFS LC-40'},
                {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'},
                {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
                {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'}],
        value='ALL',
        placeholder='Select a Launch Site',
        searchable=True
    ),
    html.Br(),
    html.Div(dcc.Graph(id='success-pie-chart')),
    html.Br(),
    html.P("Payload range (Kg):"),
    dcc.RangeSlider(
        id='payload-slider',
        min=min_payload,
        max=max_payload,
        step=1000,
        value=[min_payload, max_payload],
        marks={i: str(i) for i in range(0, int(max_payload)+1, 1000)}
    ),
    html.Br(),
    html.Div(dcc.Graph(id='success-payload-scatter-chart')),
])

@app.callback(
    Output('success-pie-chart', 'figure'),
    Input('site-dropdown', 'value')
)
def get_pie_chart(entered_site):
    if entered_site == 'ALL':
        fig = px.pie(df, values='class', names='Launch Site', title='Total Success Launches By Site')
    else:
        filtered_df = df[df['Launch Site'] == entered_site]
        counts = filtered_df['class'].value_counts().reset_index()
        counts.columns = ['class', 'count']
        fig = px.pie(
            counts, 
            values='count', 
            names='class', 
            title=f'Success vs Failure for {entered_site}',
            color='class'
        )
    return fig

@app.callback(
    Output('success-payload-scatter-chart', 'figure'),
    [Input('site-dropdown', 'value'),
     Input('payload-slider', 'value')]
)
def get_scatter_chart(entered_site, payload_range):
    if entered_site != 'ALL':
        filtered_df = df[df['Launch Site'] == entered_site]
    else:
        filtered_df = df
    
    if payload_range is not None:
        filtered_df = filtered_df[
            (filtered_df['Payload Mass (kg)'] >= payload_range[0]) & 
            (filtered_df['Payload Mass (kg)'] <= payload_range[1])
        ]
    
    fig = px.scatter(
        filtered_df,
        x='Payload Mass (kg)',
        y='class',
        color='Booster Version Category',
        title=f'Payload vs. Outcome for {entered_site}'
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)