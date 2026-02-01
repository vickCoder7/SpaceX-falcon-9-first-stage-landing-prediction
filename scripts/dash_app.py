from dash import dcc, Output, Input, html, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
from dash.dependencies import State

# Load Dataset
df = pd.read_csv('./data/spacex_launch_dash.csv')
df_geo = pd.read_csv('./data/spacex_launch_geo.csv')
max_payload = df['Payload Mass (kg)'].max()
min_payload = df['Payload Mass (kg)'].min()

# Initialize App with Dark Theme
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Load Model
model_path = './data/model.pkl'
try:
    with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)
        model = artifacts['model']
        scaler = artifacts['scaler']
        feature_names = artifacts['feature_names']
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = []

# Define Components
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1('SpaceX Falcon 9 Mission Dashboard', 
                        className='text-center text-primary mb-4'), width=12)
    ]),

    dbc.Row([
        dbc.Col(dbc.Tabs([
            # Tab 1: Historical Analysis (Existing Functionality)
            dbc.Tab(label='Historical Analysis', children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Launch Site:"),
                        dcc.Dropdown(
                            id='site-dropdown',
                            options=[{'label': 'All Sites', 'value': 'ALL'}] + 
                                    [{'label': site, 'value': site} for site in df['Launch Site'].unique()],
                            value='ALL',
                            placeholder='Select a Launch Site',
                            searchable=True,
                            style={'color': '#000'}  # Fix for dark mode dropdown visibility
                        )
                    ], width=12, className='mb-3')
                ]),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='success-pie-chart'), width=6),
                    dbc.Col(dcc.Graph(id='success-payload-scatter-chart'), width=6)
                ]),

                dbc.Row([
                    dbc.Col([
                        html.Label("Payload Range (Kg):"),
                        dcc.RangeSlider(
                            id='payload-slider',
                            min=0,
                            max=10000,
                            step=1000,
                            value=[min_payload, max_payload],
                            marks={i: str(i) for i in range(0, 10001, 2000)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], width=12, className='mt-3')
                ])
            ]),

            # Tab 2: Prediction Interface (Placeholder for ML Model)
            dbc.Tab(label='Mission Prediction', children=[
                dbc.Container([
                    html.H3("Predict Mission Outcome", className="mt-3"),
                    html.P("Enter mission parameters to estimate landing success probability.", className="text-muted"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Payload Mass (kg)"),
                            dbc.Input(id="input-payload", type="number", placeholder="e.g. 5000")
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Launch Site"),
                            dbc.Select(
                                id="input-site",
                                options=[
                                    {"label": "CCAFS LC-40", "value": "CCAFS LC-40"},
                                    {"label": "VAFB SLC-4E", "value": "VAFB SLC-4E"},
                                    {"label": "KSC LC-39A", "value": "KSC LC-39A"},
                                    {"label": "CCAFS SLC-40", "value": "CCAFS SLC-40"}
                                ]
                            )
                        ], width=4),
                         dbc.Col([
                            dbc.Label("Orbit Type"),
                            dbc.Select(
                                id="input-orbit",
                                options=[
                                    {"label": "LEO", "value": "LEO"},
                                    {"label": "ISS", "value": "ISS"},
                                    {"label": "PO", "value": "PO"},
                                    {"label": "GTO", "value": "GTO"},
                                    {"label": "ES-L1", "value": "ES-L1"},
                                    {"label": "SSO", "value": "SSO"},
                                    {"label": "HEO", "value": "HEO"},
                                    {"label": "MEO", "value": "MEO"},
                                    {"label": "VLEO", "value": "VLEO"},
                                    {"label": "SO", "value": "SO"},
                                    {"label": "GEO", "value": "GEO"}
                                ]
                            )
                        ], width=4),
                    ], className="mb-3"),
                    
                    dbc.Button("Predict Outcome", id="predict-btn", color="success", className="mb-3"),
                    
                    html.Div(id="prediction-output", className="lead text-center")
                ], fluid=True)
            ]),

            # Tab 3: Launch Site Map
            # dbc.Tab(label='Launch Map', children=[
            #     dbc.Row([
            #         dbc.Col(html.H3("Launch Site Locations", className="mt-3 text-center"), width=12),
            #         dbc.Col(html.P("Interactive map of launch sites and mission outcomes.", className="text-muted text-center"), width=12)
            #     ]),
            #     dbc.Row([
            #         dbc.Col(dcc.Graph(id='launch-map'), width=12)
            #     ])
            # ])
        ]))
    ])
], fluid=True)

# Callback for Pie Chart
@app.callback(
    Output('success-pie-chart', 'figure'),
    Input('site-dropdown', 'value')
)
def get_pie_chart(entered_site):
    if entered_site == 'ALL':
        fig = px.pie(df, values='class', names='Launch Site', title='Total Success Launches By Site', template='plotly_dark')
    else:
        filtered_df = df[df['Launch Site'] == entered_site]
        counts = filtered_df['class'].value_counts().reset_index()
        counts.columns = ['class', 'count']
        fig = px.pie(counts, values='count', names='class', 
                     title=f'Success vs Failure for {entered_site}', 
                     color='class', template='plotly_dark')
    return fig

# Callback for Scatter Chart
@app.callback(
    Output('success-payload-scatter-chart', 'figure'),
    [Input('site-dropdown', 'value'),
     Input('payload-slider', 'value')]
)
def get_scatter_chart(entered_site, payload_range):
    low, high = payload_range
    filtered_df = df[(df['Payload Mass (kg)'] >= low) & (df['Payload Mass (kg)'] <= high)]
    
    if entered_site != 'ALL':
        filtered_df = filtered_df[filtered_df['Launch Site'] == entered_site]
    
    fig = px.scatter(
        filtered_df, x='Payload Mass (kg)', y='class', 
        color='Booster Version Category',
        title=f'Payload vs. Outcome for {entered_site}',
        template='plotly_dark'
    )
    return fig

# Callback for Launch Map
# @app.callback(
#     Output('launch-map', 'figure'),
#     Input('site-dropdown', 'value')
# )
# def get_map(entered_site):
#     if entered_site == 'ALL':
#         filtered_df = df_geo
#         title = 'All Launch Sites'
#     else:
#         filtered_df = df_geo[df_geo['Launch Site'] == entered_site]
#         title = f'Launch Site: {entered_site}'
    
#     fig = px.scatter_mapbox(
#         filtered_df,
#         lat='Lat',
#         lon='Long',
#         color='class',
#         color_continuous_scale=['red', 'green'],
#         size_max=15,
#         zoom=3,
#         hover_name='Launch Site',
#         hover_data=['Payload', 'Booster Version', 'class'],
#         title=title,
#         mapbox_style='carto-darkmatter',
#         template='plotly_dark'
#     )
#     # Adjust zoom if specific site is selected
#     if entered_site != 'ALL':
#         fig.update_layout(mapbox_zoom=10)
        
#     fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
#     return fig

# Callback for Prediction
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State("input-payload", "value"),
     State("input-site", "value"),
     State("input-orbit", "value")]
)
def predict_landing(n_clicks, payload, site, orbit):
    if n_clicks is None:
        return ""
    
    if not model:
        return dbc.Alert("Model not loaded!", color="danger")
        
    if not payload or not site or not orbit:
        return dbc.Alert("Please fill in all fields.", color="warning")

    try:
        # Create input vector
        input_data = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Set User Inputs
        input_data['PayloadMass'] = float(payload)
        
        if f'Orbit_{orbit}' in feature_names:
            input_data[f'Orbit_{orbit}'] = 1
            
        if f'LaunchSite_{site}' in feature_names:
            input_data[f'LaunchSite_{site}'] = 1
            
        # Set Assumptions for a Modern Flight (Block 5, etc.)
        input_data['FlightNumber'] = 150 # Future flight
        input_data['Flights'] = 1
        input_data['GridFins'] = 1 # True
        input_data['Reused'] = 0 # New booster assumption
        input_data['Legs'] = 1 # True
        input_data['Block'] = 5
        input_data['ReusedCount'] = 0
        
        # Scale
        X = scaler.transform(input_data)
        
        # Predict
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        
        if pred == 1:
            return dbc.Alert(f"SUCCESS PREDICTED (Probability: {prob:.2%})", color="success")
        else:
            return dbc.Alert(f"FAILURE PREDICTED (Probability: {prob:.2%})", color="danger")
            
    except Exception as e:
        return dbc.Alert(f"Error during prediction: {str(e)}", color="danger")

if __name__ == '__main__':
    app.run(debug=True)