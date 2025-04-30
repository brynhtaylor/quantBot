import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objs as go
import pandas as pd
import os
from dash.exceptions import PreventUpdate

from agent_strategy_service import AgentService
from backtest_service import BacktestService
from agent_interpretation_service import StrategyInterpreter

# === Config ===
DATA_DIR = "data"
CLUSTERED_PATH = os.path.join(DATA_DIR, "clustered", "clustered_financial_data.csv")
MERGED_PATH = os.path.join(DATA_DIR, "processed", "merged_financial_data.csv")
CRSP_PATH = os.path.join(DATA_DIR, "raw", "crsp.csv")
KMEANS_STATS_PATH = os.path.join(DATA_DIR, "clustered", "kmeans_cluster_stats.csv")

def load_benchmark(crsp_path, strategy_periods):
    crsp = pd.read_csv(crsp_path, low_memory=False)
    crsp['date'] = pd.to_datetime(crsp['date'])
    crsp['year_month'] = crsp['date'].dt.year * 100 + crsp['date'].dt.month
    monthly_returns = crsp.groupby('year_month')['sprtrn'].mean().sort_index()
    benchmark_cumulative = (1 + monthly_returns).cumprod()
    return benchmark_cumulative.reindex(strategy_periods, method='ffill')

def yyyymm_to_datetime(yyyymm_int):
    return pd.to_datetime(yyyymm_int.astype(str), format='%Y%m')

# === Initialize Services ===
def get_available_models():
    # Placeholder static list (could dynamically fetch installed models later)
    return ["gemma3", "deepseek-r1", "mistral"]

# === Dash App Setup ===
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "quantBot - Agentic Trading Strategy Generator"

# Define color scheme
colors = {
    'background': '#121212',
    'card': '#1e1e1e',
    'primary': '#00c853',  # Terminal green
    'secondary': '#bb86fc',
    'text': '#e0e0e0',
    'muted_text': '#9e9e9e',
    'border': '#333333',
    'positive': '#00c853',
    'negative': '#ff5252',
    'neutral': '#64b5f6'
}

# === Layout ===
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.I(className="fas fa-robot", style={"marginRight": "10px", "fontSize": "24px", "color": colors['primary']}),
            html.H1("quantBot", style={"margin": "0", "fontFamily": "'Inter', sans-serif", "letterSpacing": "0.5px", "color": colors['primary']})
        ], style={"display": "flex", "alignItems": "center"}),
        
        html.Div([
            html.P("v1.0.3", style={"margin": "0", "color": colors['muted_text'], "fontFamily": "'JetBrains Mono', monospace"})
        ])
    ], style={
        "padding": "20px",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "borderBottom": f"1px solid {colors['border']}",
        "backgroundColor": colors['card'],
    }),
    
    # Main content area
    html.Div([
        # Initial centered input area (visible at first)
        html.Div([
            html.Div([
                html.Div([
                    html.I(className="fas fa-terminal", style={"fontSize": "40px", "color": colors['primary'], "marginBottom": "20px"}),
                    html.H2("Generate Trading Strategies with AI", style={"textAlign": "center", "marginBottom": "30px"})
                ], style={"textAlign": "center"}),
                
                html.Div([
                    html.Label("Tell me about your trading strategy criteria:", style={"marginBottom": "10px", "fontWeight": "bold"}),
                    dcc.Textarea(
                        id="user-prompt",
                        placeholder="e.g., Generate a momentum strategy that focuses on low volatility stocks during bear markets",
                        style={
                            "width": "100%", 
                            "height": "150px", 
                            "padding": "15px",
                            "backgroundColor": colors['background'], 
                            "color": colors['text'],
                            "border": f"1px solid {colors['border']}",
                            "borderRadius": "4px",
                            "fontFamily": "'JetBrains Mono', monospace",
                            "fontSize": "14px",
                            "resize": "none"
                        }
                    ),
                    
                    html.Div([
                        html.Label("Choose AI Model Engine:", style={"marginBottom": "10px", "fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="model-selector",
                            options=[{"label": m, "value": m} for m in get_available_models()],
                            value="gemma3",
                            style={
                                "backgroundColor": colors['background'],
                                "color": colors['text']
                            }
                        ),
                    ], style={"marginTop": "20px"}),
                    
                    html.Button(
                        [
                            html.I(className="fas fa-cogs", style={"marginRight": "10px"}),
                            "Generate Strategies"
                        ],
                        id="run-agent",
                        n_clicks=0,
                        style={
                            "width": "100%",
                            "padding": "12px",
                            "marginTop": "25px",
                            "backgroundColor": colors['primary'],
                            "color": colors['background'],
                            "border": "none",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                            "fontWeight": "bold",
                            "fontSize": "16px",
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center"
                        }
                    ),
                    
                    html.Div(id="loading-container", children=[
                        dcc.Loading(
                            id="loading",
                            type="default",
                            children=html.Div(id="loading-output")
                        )
                    ], style={"marginTop": "20px", "textAlign": "center"})
                ], style={"width": "100%"})
            ], style={
                "width": "600px",
                "padding": "30px",
                "backgroundColor": colors['card'],
                "borderRadius": "8px",
                "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                "margin": "auto"
            })
        ], id="central-input-container", style={
            "display": "flex",
            "justifyContent": "center",
            "alignItems": "center",
            "minHeight": "calc(100vh - 100px)",
            "transition": "all 0.5s ease-in-out"
        }),
        
        # Results area (hidden initially)
        html.Div([
            # Left sidebar with input
            html.Div([
                html.Div([
                    html.Label("Refine Your Strategy:", style={"fontWeight": "bold", "marginBottom": "10px"}),
                    dcc.Textarea(
                        id="user-prompt-sidebar",
                        style={
                            "width": "100%", 
                            "height": "150px", 
                            "padding": "10px",
                            "backgroundColor": colors['background'], 
                            "color": colors['text'],
                            "border": f"1px solid {colors['border']}",
                            "borderRadius": "4px",
                            "fontFamily": "'JetBrains Mono', monospace",
                            "fontSize": "14px",
                            "resize": "none"
                        }
                    ),
                    
                    html.Div([
                        html.Label("Model:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                        dcc.Dropdown(
                            id="model-selector-sidebar",
                            options=[{"label": m, "value": m} for m in get_available_models()],
                            value="gemma3",
                            style={
                                "backgroundColor": colors['background'],
                                "color": colors['text']
                            }
                        ),
                    ], style={"marginTop": "20px"}),
                    
                    html.Button(
                        [
                            html.I(className="fas fa-sync-alt", style={"marginRight": "10px"}),
                            "Regenerate"
                        ],
                        id="run-agent-sidebar",
                        n_clicks=0,
                        style={
                            "width": "100%",
                            "padding": "10px",
                            "marginTop": "20px",
                            "backgroundColor": colors['primary'],
                            "color": colors['background'],
                            "border": "none",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                            "fontWeight": "bold",
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center"
                        }
                    ),
                    
                    html.Hr(style={"margin": "30px 0", "borderColor": colors['border']}),
                    
                    html.Div([
                        html.H4("System Status", style={"marginBottom": "15px"}),
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-server", style={"marginRight": "10px", "color": colors['primary']}),
                                "Model Active"
                            ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
                            html.Div([
                                html.I(className="fas fa-database", style={"marginRight": "10px", "color": colors['primary']}),
                                "Data Connected"
                            ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
                            html.Div([
                                html.I(className="fas fa-chart-line", style={"marginRight": "10px", "color": colors['primary']}),
                                "Analytics Ready"
                            ], style={"display": "flex", "alignItems": "center"})
                        ])
                    ]),
                    
                    # Loading indicator for sidebar
                    html.Div(id="loading-sidebar-container", children=[
                        dcc.Loading(
                            id="loading-sidebar",
                            type="default",
                            children=html.Div(id="loading-output-sidebar")
                        )
                    ], style={"marginTop": "20px", "textAlign": "center"})
                ], style={"padding": "20px"})
            ], style={
                "width": "300px",
                "backgroundColor": colors['card'],
                "borderRadius": "8px",
                "marginRight": "20px",
                "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"
            }),
            
            # Main results area
            html.Div([
                # Tabs for different views
                dcc.Tabs(
                    id="result-tabs",
                    value="tab-performance",
                    children=[
                        dcc.Tab(
                            label="Performance",
                            value="tab-performance",
                            style={
                                "backgroundColor": colors['background'],
                                "color": colors['muted_text'],
                                "border": f"1px solid {colors['border']}",
                                "fontFamily": "'Inter', sans-serif",
                                "padding": "15px"
                            },
                            selected_style={
                                "backgroundColor": colors['card'],
                                "color": colors['primary'],
                                "border": f"1px solid {colors['border']}",
                                "borderBottom": f"2px solid {colors['primary']}",
                                "fontFamily": "'Inter', sans-serif",
                                "padding": "15px"
                            }
                        ),
                        dcc.Tab(
                            label="Strategy Details",
                            value="tab-details",
                            style={
                                "backgroundColor": colors['background'],
                                "color": colors['muted_text'],
                                "border": f"1px solid {colors['border']}",
                                "fontFamily": "'Inter', sans-serif",
                                "padding": "15px"
                            },
                            selected_style={
                                "backgroundColor": colors['card'],
                                "color": colors['primary'],
                                "border": f"1px solid {colors['border']}",
                                "borderBottom": f"2px solid {colors['primary']}",
                                "fontFamily": "'Inter', sans-serif",
                                "padding": "15px"
                            }
                        ),
                    ],
                    style={
                        "marginBottom": "20px"
                    }
                ),
                
                # Performance Tab Content
                html.Div([
                    dcc.Graph(
                        id="strategy-returns-graph",
                        config={"displayModeBar": True, "scrollZoom": True},
                        style={"height": "500px"}
                    ),
                    
                    # Performance metrics cards
                    html.Div([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-chart-line", style={"fontSize": "24px", "color": colors['positive'], "marginBottom": "10px"}),
                                html.H4("Best Strategy", style={"margin": "0", "marginBottom": "5px"}),
                                html.Div(id="best-strategy-name", style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "5px"}),
                                html.Div(id="best-strategy-sharpe", style={"fontSize": "24px", "color": colors['positive']})
                            ], style={
                                "padding": "20px",
                                "backgroundColor": colors['card'],
                                "borderRadius": "8px",
                                "textAlign": "center",
                                "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
                            }),
                            
                            html.Div([
                                html.I(className="fas fa-chart-pie", style={"fontSize": "24px", "color": colors['neutral'], "marginBottom": "10px"}),
                                html.H4("Avg Hit Rate", style={"margin": "0", "marginBottom": "5px"}),
                                html.Div(id="avg-hit-rate", style={"fontSize": "24px", "color": colors['neutral']})
                            ], style={
                                "padding": "20px",
                                "backgroundColor": colors['card'],
                                "borderRadius": "8px",
                                "textAlign": "center",
                                "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
                                "marginTop": "15px"
                            }),
                        ], style={"flex": "1", "marginRight": "15px"}),
                        
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-arrow-trend-up", style={"fontSize": "24px", "color": colors['secondary'], "marginBottom": "10px"}),
                                html.H4("Cumulative Return", style={"margin": "0", "marginBottom": "5px"}),
                                html.Div(id="best-strategy-return", style={"fontSize": "24px", "color": colors['secondary']})
                            ], style={
                                "padding": "20px",
                                "backgroundColor": colors['card'],
                                "borderRadius": "8px",
                                "textAlign": "center",
                                "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
                            }),
                            
                            html.Div([
                                html.I(className="fas fa-arrow-trend-down", style={"fontSize": "24px", "color": colors['negative'], "marginBottom": "10px"}),
                                html.H4("Max Drawdown", style={"margin": "0", "marginBottom": "5px"}),
                                html.Div(id="avg-max-drawdown", style={"fontSize": "24px", "color": colors['negative']})
                            ], style={
                                "padding": "20px",
                                "backgroundColor": colors['card'],
                                "borderRadius": "8px",
                                "textAlign": "center", 
                                "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
                                "marginTop": "15px"
                            }),
                        ], style={"flex": "1"})
                    ], style={"display": "flex", "marginTop": "20px"})
                ], id="tab-performance-content"),
                
                # Strategy Details Tab Content
                html.Div([
                    # Strategy table wrapped in a container for styling
                    html.Div([
                        html.H3("Trading Strategies", style={"marginBottom": "15px"}),
                        dash_table.DataTable(
                            id="strategy-table",
                            style_header={
                                'backgroundColor': colors['background'],
                                'color': colors['primary'],
                                'fontWeight': 'bold',
                                'border': f"1px solid {colors['border']}",
                                'fontFamily': "'Inter', sans-serif"
                            },
                            style_cell={
                                'backgroundColor': colors['card'],
                                'color': colors['text'],
                                'border': f"1px solid {colors['border']}",
                                'fontFamily': "'JetBrains Mono', monospace",
                                'fontSize': '14px',
                                'padding': '10px',
                                'textAlign': 'left'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'column_id': 'Sharpe'},
                                    'color': colors['positive']
                                },
                                {
                                    'if': {'column_id': 'Max_Drawdown'},
                                    'color': colors['negative']
                                },
                                {
                                    'if': {'column_id': 'Hit_Rate'},
                                    'color': colors['neutral']
                                },
                            ],
                            style_table={
                                'overflowX': 'auto',
                                'border': f"1px solid {colors['border']}"
                            },
                            page_size=10,
                            sort_action='native',
                            filter_action='native',
                        ),
                    ], style={
                        "backgroundColor": colors['card'],
                        "padding": "20px",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
                    }),
                    
                    # Strategy interpretation from AI
                    html.Div([
                        html.H3("AI Strategy Interpretation", style={"marginBottom": "15px", "display": "flex", "alignItems": "center"}),
                        html.Div(id="strategy-interpretation", children=[
                            html.P("Generate strategies to see AI interpretation of results.", style={"color": colors['muted_text']})
                        ], style={
                            "fontFamily": "'Inter', sans-serif",
                            "lineHeight": "1.6"
                        })
                    ], style={
                        "backgroundColor": colors['card'],
                        "padding": "20px",
                        "borderRadius": "8px",
                        "marginTop": "20px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
                    })
                ], id="tab-details-content", style={"display": "none"})
            ], style={"flex": "1"})
        ], id="results-container", style={
            "display": "none",
            "padding": "20px",
            "flexDirection": "row"
        })
    ], style={
        "backgroundColor": colors['background'],
        "color": colors['text'],
        "minHeight": "100vh",
        "fontFamily": "'Inter', sans-serif"
    })
], style={"backgroundColor": colors['background']})

# === Callbacks ===
@app.callback(
    [Output("central-input-container", "style"),
     Output("results-container", "style"),
     Output("user-prompt-sidebar", "value"),
     Output("model-selector-sidebar", "value"),
     Output("loading-output", "children")],
    [Input("run-agent", "n_clicks")],
    [State("user-prompt", "value"),
     State("model-selector", "value"),
     State("central-input-container", "style")]
)
def transition_layout(n_clicks, user_prompt, model_selected, current_style):
    if n_clicks == 0:
        raise PreventUpdate
    
    # Update the central container to be hidden
    new_central_style = current_style.copy()
    new_central_style["display"] = "none"
    
    # Show the results container
    results_style = {
        "display": "flex",
        "padding": "20px",
        "flexDirection": "row"
    }
    
    # Copy the values to the sidebar
    return new_central_style, results_style, user_prompt, model_selected, None

@app.callback(
    [Output("tab-performance-content", "style"),
     Output("tab-details-content", "style")],
    [Input("result-tabs", "value")]
)
def render_tab_content(tab):
    if tab == "tab-performance":
        return {"display": "block"}, {"display": "none"}
    else:
        return {"display": "none"}, {"display": "block"}

@app.callback(
    [Output("strategy-returns-graph", "figure"),
     Output("strategy-table", "data"),
     Output("strategy-table", "columns"),
     Output("best-strategy-name", "children"),
     Output("best-strategy-sharpe", "children"),
     Output("best-strategy-return", "children"),
     Output("avg-hit-rate", "children"),
     Output("avg-max-drawdown", "children"),
     Output("strategy-interpretation", "children"),
     Output("loading-output-sidebar", "children")],
    [Input("run-agent", "n_clicks"),
     Input("run-agent-sidebar", "n_clicks")],
    [State("user-prompt", "value"),
     State("model-selector", "value"),
     State("user-prompt-sidebar", "value"),
     State("model-selector-sidebar", "value")]
)
def run_agent_and_backtest(main_clicks, sidebar_clicks, main_prompt, main_model, sidebar_prompt, sidebar_model):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Determine which inputs to use based on which button was clicked
    if trigger_id == "run-agent":
        if main_clicks == 0:
            raise PreventUpdate
        user_prompt = main_prompt
        model_selected = main_model
    else:  # run-agent-sidebar
        if sidebar_clicks == 0:
            raise PreventUpdate
        user_prompt = sidebar_prompt
        model_selected = sidebar_model

    try:
        # === 1. Run Agent
        agent = AgentService(model_name=model_selected)
        strategies = agent.generate_strategies(user_guidance=user_prompt or "", kmeans_stats_path=KMEANS_STATS_PATH)

        # === 2. Backtest
        backtester = BacktestService(
            clustered_path=CLUSTERED_PATH,
            merged_path=MERGED_PATH,
            crsp_path=CRSP_PATH
        )
        results_df, metrics_df = backtester.backtest_strategies(strategies)

        # === 3. Plot Results
        fig = go.Figure()

        # Custom color palette for strategies
        strategy_colors = [
            '#00c853',  # Primary green
            '#bb86fc',  # Purple
            '#03dac6',  # Teal
            '#ff9e80',  # Orange
            '#b388ff',  # Lavender
            '#8c9eff',  # Light blue
            '#ea80fc',  # Pink
            '#ffd180',  # Amber
        ]
        
        color_idx = 0
        for name, grp in results_df.groupby('strategy_name'):
            fig.add_trace(go.Scatter(
                x=yyyymm_to_datetime(grp['year_month']),
                y=(grp['cumulative_return'] - 1) * 100,
                mode='lines',
                name=name,
                line=dict(color=strategy_colors[color_idx % len(strategy_colors)], width=2)
            ))
            color_idx += 1

        strategy_periods = results_df['year_month'].sort_values().unique()
        benchmark_cumulative = load_benchmark(CRSP_PATH, strategy_periods)

        fig.add_trace(go.Scatter(
            x=yyyymm_to_datetime(strategy_periods),
            y=(benchmark_cumulative.values - 1) * 100,
            mode='lines',
            name='Passive Benchmark (S&P 500)',
            line=dict(dash='dash', color='#e0e0e0', width=1.5)
        ))

        # Apply a modern, dark-themed layout to the chart
        fig.update_layout(
            title={
                'text': "Cumulative Returns (%) by Strategy vs Benchmark",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20, color=colors['text'])
            },
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            template="plotly_dark",
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text'], family="'Inter', sans-serif"),
            legend=dict(
                bgcolor=colors['card'],
                bordercolor=colors['border'],
                borderwidth=1,
                font=dict(color=colors['text'])
            ),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=80, b=40),
            xaxis=dict(
                showgrid=True,
                gridcolor=colors['border'],
                zeroline=False,
                linecolor=colors['border']
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=colors['border'],
                zeroline=False,
                linecolor=colors['border']
            )
        )

        # === 4. Strategy Table (metrics enriched)
        strategies_df = pd.DataFrame(strategies)
        enriched_df = pd.merge(strategies_df, metrics_df, on='strategy_name', how='left')

        # Format nicely
        enriched_df["Sharpe"] = enriched_df["Sharpe"].round(2)
        enriched_df["Max_Drawdown"] = (enriched_df["Max_Drawdown"] * 100).round(1)
        enriched_df["Hit_Rate"] = (enriched_df["Hit_Rate"] * 100).round(1)

        display_columns = [
            "strategy_name", "regime", "long_feature", "short_feature", 
            "Sharpe", "Max_Drawdown", "Hit_Rate"
        ]

        # Only keep columns that exist
        enriched_df = enriched_df[[col for col in display_columns if col in enriched_df.columns]]

        # Rebuild table spec
        columns = [{"name": col.replace('_', ' '), "id": col} for col in enriched_df.columns]
        table_data = enriched_df.to_dict("records")  # no extra slicing

        # Optional sanity checks
        assert all(isinstance(row, dict) for row in table_data), "Malformed table_data"
        assert all("id" in col for col in columns), "Missing id in columns"
        
        # === 5. Summary statistics for cards
        best_strategy = enriched_df.loc[enriched_df['Sharpe'].idxmax()] if not enriched_df.empty else None
        best_strategy_name = best_strategy['strategy_name'] if best_strategy is not None else "N/A"
        best_strategy_sharpe = f"{best_strategy['Sharpe']:.2f}" if best_strategy is not None else "N/A"
        
        # Calculate best return from results_df
        if not results_df.empty:
            final_returns = results_df.groupby('strategy_name')['cumulative_return'].last()
            best_return = f"{(max(final_returns) - 1) * 100:.1f}%"
        else:
            best_return = "N/A"
            
        avg_hit_rate = f"{enriched_df['Hit_Rate'].mean():.1f}%" if not enriched_df.empty else "N/A"
        avg_max_drawdown = f"{enriched_df['Max_Drawdown'].mean():.1f}%" if not enriched_df.empty else "N/A"
        
        # === 6. Generate AI interpretation
        interpreter = StrategyInterpreter(model_name=model_selected)
        interpretation_text = interpreter.interpret_strategies(strategies, metrics_df, user_guidance=user_prompt)

        interpretation = dcc.Markdown(interpretation_text, style={"whiteSpace": "pre-wrap"})

        return fig, table_data, columns, best_strategy_name, best_strategy_sharpe, best_return, avg_hit_rate, avg_max_drawdown, interpretation, None

    except Exception as e:
        print("Error occurred during agent generation or backtesting:", e)
        
        # Create empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title={
                'text': "Error Generating Strategies",
                'y': 0.5,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'middle',
                'font': dict(size=20, color=colors['negative'])
            },
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text'], family="'Inter', sans-serif"),
            annotations=[
                dict(
                    text=f"Error: {str(e)}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.4,
                    showarrow=False,
                    font=dict(size=14, color=colors['muted_text'])
                ),
                dict(
                    text="Please try again with different parameters",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.3,
                    showarrow=False,
                    font=dict(size=14, color=colors['muted_text'])
                )
            ]
        )
        
        error_interpretation = [
            html.Div([
                html.I(className="fas fa-exclamation-triangle", style={"color": colors['negative'], "fontSize": "24px", "marginRight": "10px"}),
                html.Span("Error Processing Request", style={"color": colors['negative'], "fontSize": "18px"})
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "15px"}),
            html.P(f"Details: {str(e)}", style={"color": colors['muted_text']}),
            html.P("Suggestions:", style={"fontWeight": "bold", "marginTop": "15px"}),
            html.Ul([
                html.Li("Check your prompt for specific instructions"),
                html.Li("Try a different AI model"),
                html.Li("Ensure data files are properly located")
            ])
        ]
        
        return fig, [], [], "Error", "N/A", "N/A", "N/A", "N/A", error_interpretation, None

# Callback to sync the main input with sidebar input when transitioning
@app.callback(
    Output("central-input-container", "style", allow_duplicate=True),
    [Input("run-agent-sidebar", "n_clicks")],
    [State("user-prompt-sidebar", "value"),
     State("model-selector-sidebar", "value")],
    prevent_initial_call=True
)
def return_to_input(n_clicks, prompt, model):
    if n_clicks > 0:
        # Keep in results view
        return {"display": "none"}
    raise PreventUpdate

# Add client-side callback for tab switching
app.clientside_callback(
    """
    function(tabValue) {
        if (tabValue === 'tab-performance') {
            return {'display': 'block'}, {'display': 'none'};
        } else {
            return {'display': 'none'}, {'display': 'block'};
        }
    }
    """,
    [Output("tab-performance-content", "style", allow_duplicate=True),
     Output("tab-details-content", "style", allow_duplicate=True)],
    [Input("result-tabs", "value")],
    prevent_initial_call=True
)

# === Custom CSS ===
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #1e1e1e;
            }
            ::-webkit-scrollbar-thumb {
                background: #444;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #00c853;
            }
            
            /* Dropdown styling */
            .Select-control {
                background-color: #1e1e1e !important;
                border-color: #333 !important;
                color: #e0e0e0 !important;
            }
            .Select-menu-outer {
                background-color: #1e1e1e !important;
                border: 1px solid #333 !important;
            }
            .Select-value-label {
                color: #e0e0e0 !important;
            }
            .Select-menu-outer .Select-option {
                background-color: #1e1e1e !important;
                color: #e0e0e0 !important;
            }
            .Select-menu-outer .Select-option:hover {
                background-color: #333 !important;
            }
            
            /* Button hover effects */
            button:hover {
                filter: brightness(1.1);
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                transition: all 0.2s ease;
            }
            
            /* Loading spinner styles */
            ._dash-loading {
                margin: auto;
                color: #00c853 !important;
            }
            
            /* Table row hover effect */
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table tr:hover {
                background-color: #2c2c2c !important;
            }
            
            /* Animated terminal cursor effect for the header */
            @keyframes blink {
                0%, 100% { opacity: 1; }
                50% { opacity: 0; }
            }
            
            .terminal-cursor::after {
                content: '|';
                margin-left: 3px;
                animation: blink 1s infinite;
                color: #00c853;
            }
            
            /* Tab transitions */
            .tab-content {
                transition: opacity 0.3s ease-in-out;
            }
            
            /* Card hover effect */
            .metric-card {
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .metric-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# === Run App ===
if __name__ == "__main__":
    app.run(debug=True)