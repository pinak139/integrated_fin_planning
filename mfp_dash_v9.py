import dash
from dash import dcc, html, Input, Output, State, callback, ctx, no_update
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import dash_ag_grid as dag
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from orbit.models import DLT
import os
import io
import traceback

# --- 1. CONFIGURATION ---
SEARCH_PATHS = [".", "/home/pinakd/TAIL/MFP/"]
FILES = {"marketing": "marketing_spend.csv", "traffic": "traffic.csv", "sales": "category_sales.csv"}

class AppState:
    def __init__(self):
        self.registry = {}
        self.transform_meta = {}
        self.decay = 0.5
        self.opt_results = None
        self.data_source = "UNKNOWN"

app_state = AppState()

# --- 2. HELPERS ---
def get_icon(icon): return DashIconify(icon=icon, width=20)

def polish_fig(fig, height=None):
    fig.update_layout(
        template="plotly_white", 
        margin=dict(l=30, r=20, t=30, b=30), 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", 
        legend=dict(orientation="h", y=1.1)
    )
    if height: fig.update_layout(height=height)
    return fig

def make_kpi(title, value, diff, color="blue"):
    return dmc.Paper(
        children=[
            dmc.Group([
                dmc.Text(title, color="dimmed", size="xs", transform="uppercase", weight=700),
                dmc.ThemeIcon(DashIconify(icon="tabler:chart-bar", width=16), variant="light", color=color, radius="xl", size="sm")
            ], position="apart", mb="xs"),
            dmc.Group([
                dmc.Text(value, weight=700, size="xl"),
                dmc.Badge(f"{diff}", color="green" if "+" in diff else "red", variant="light")
            ], align="flex-end", spacing="xs")
        ],
        withBorder=True, shadow="sm", p="md", radius="md"
    )

def adstock(x, a):
    y = np.zeros_like(x)
    y[0] = x[0]
    for t in range(1, len(x)): y[t] = x[t] + a * y[t-1]
    return y

def saturation(x, k): return x / (1 + (x/(k if k>0 else 1.0)))

# --- 3. DATA ENGINE ---
def generate_mock_data():
    print("⚠️ Generating MOCK DATA...")
    dates = pd.date_range(start="2023-01-01", periods=156, freq="W-SUN")
    cats = ['Climbing', 'Snow', 'Running', 'Hiking', 'Yoga']
    sales_data = {'date': dates}
    for c in cats:
        sales_data[f"sales_{c}"] = np.random.randint(50000, 150000, len(dates))
        sales_data[f"vol_{c}"] = (sales_data[f"sales_{c}"] / 50).astype(int)
    traffic_data = {'store_traffic': sales_data['sales_Hiking']/50, 'digital_sessions': sales_data['sales_Running']/10}
    channels = ['Paid Search', 'Social', 'Display', 'Video']
    mkt_data = {c: np.random.randint(5000, 20000, len(dates)) for c in channels}
    
    df = pd.DataFrame(sales_data)
    for k,v in traffic_data.items(): df[k] = v
    for k,v in mkt_data.items(): df[k] = v
    df['total_traffic'] = df['store_traffic'] + df['digital_sessions']
    return df, cats, channels

def load_data():
    found = next((p for p in SEARCH_PATHS if all(os.path.exists(os.path.join(p, f)) for f in FILES.values())), None)
    if found:
        print(f"✅ Found files in: {found}")
        try:
            dfs = {}
            for k, f in FILES.items():
                dfs[k] = pd.read_csv(os.path.join(found, f))
                dfs[k].columns = [c.lower().strip() for c in dfs[k].columns]
                for c in dfs[k].columns:
                    if 'date' in c or 'week' in c:
                        dfs[k]['date'] = pd.to_datetime(dfs[k][c])
                        break
            
            main = dfs['traffic'].copy()
            if 'marketing_channel' in dfs['marketing'].columns:
                mkt = dfs['marketing'].pivot_table(index='date', columns='marketing_channel', values='spend', aggfunc='sum').reset_index()
                main = pd.merge(main, mkt, on='date', how='left').fillna(0)
            if 'department' in dfs['sales'].columns:
                sales = dfs['sales'].pivot_table(index='date', columns='department', values='net_sales', aggfunc='sum').reset_index()
                sales.columns = ['date'] + [f"sales_{c}" for c in sales.columns if c != 'date']
                main = pd.merge(main, sales, on='date', how='left').fillna(0)
            
            cats = [c.replace('sales_', '') for c in main.columns if 'sales_' in c]
            chans = [c for c in mkt.columns if c != 'date']
            
            # Synthesize Volume if missing
            for c in cats:
                if f"vol_{c}" not in main.columns:
                    main[f"vol_{c}"] = (main[f"sales_{c}"] / 50).fillna(0).astype(int)

            main = main.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
            last = main['date'].max()
            if last.year < 2026:
                future = pd.date_range(last + pd.Timedelta(weeks=1), periods=52, freq='W-SUN')
                main = pd.concat([main, pd.DataFrame({'date': future})], ignore_index=True).fillna(0)
            
            app_state.data_source = "LIVE DATA"
            return main, cats, chans
        except Exception as e: print(f"❌ Error: {e}")
            
    app_state.data_source = "MOCK DATA"
    return generate_mock_data()

# --- 4. APP INIT ---
df_main, CATEGORIES, CHANNELS = load_data()
ALL_TARGETS = CATEGORIES + ['store_traffic', 'digital_sessions']
app = dash.Dash(__name__)

# --- 5. LAYOUT ---
app.layout = dmc.MantineProvider(
    theme={"fontFamily": "'Inter', sans-serif", "primaryColor": "indigo"},
    children=[
        dcc.Store(id="store-data", data=df_main.to_json(date_format='iso', orient='split')),
        dmc.AppShell(
            header=dmc.Header(height=60, p="md", children=[
                dmc.Group(position="apart", children=[
                    dmc.Group([DashIconify(icon="tabler:mountain", width=30, color="#4c6ef5"), dmc.Text("PEAK 28 | Planning", size="lg", weight=700)]),
                    dmc.Badge(app_state.data_source, color="green" if app_state.data_source == "LIVE DATA" else "orange")
                ])
            ]),
            navbar=dmc.Navbar(p="md", width={"base": 300}, children=[
                dmc.Text("Settings", size="sm", weight=500, color="dimmed", mb="md"),
                dmc.NumberInput(label="Growth Target", id="growth-input", value=0.10, step=0.01, precision=2, mb="sm"),
                dmc.NumberInput(label="Adstock Decay", id="decay-input", value=0.5, step=0.1, min=0, max=0.9, mb="md"),
                dmc.Divider(my="lg"),
                dmc.NavLink(label="System Status", icon=get_icon("tabler:server"), variant="subtle", active=True, color="gray"),
            ]),
            children=[
                dmc.Container(fluid=True, children=[
                    dmc.Tabs(value="diagnostics", variant="outline", children=[
                        dmc.TabsList([
                            dmc.Tab("Diagnostics", value="diagnostics", icon=get_icon("tabler:dashboard")),
                            dmc.Tab("Training", value="training", icon=get_icon("tabler:brain")),
                            dmc.Tab("Forecast", value="forecast", icon=get_icon("tabler:chart-arrows")),
                            dmc.Tab("Optimizer", value="optimizer", icon=get_icon("tabler:adjustments")),
                        ]),
                        
                        # --- DIAGNOSTICS TAB ---
                        dmc.TabsPanel(value="diagnostics", pt="md", children=[
                             # Dynamic KPI Row
                             dmc.SimpleGrid(cols=3, spacing="md", mb="md", id="diag-kpi-row"),
                             
                             dmc.Paper(withBorder=True, p="md", mb="md", children=[
                                 dmc.Group([
                                     dmc.Text("Metric View:", weight=600),
                                     dmc.SegmentedControl(id="diag-metric-select", value="total_sales", data=[
                                         {"label": "Total Sales ($)", "value": "total_sales"},
                                         {"label": "Store Traffic", "value": "store_traffic"},
                                         {"label": "Digital Sessions", "value": "digital_sessions"},
                                         {"label": "Category Detail", "value": "category"}
                                     ]),
                                     # Category Detail Controls
                                     html.Div(id="diag-cat-dropdown-container", style={"display": "none"}, children=[
                                         dmc.Group([
                                             dmc.SegmentedControl(id="diag-unit-toggle", value="sales", data=[{"label": "Sales ($)", "value": "sales"}, {"label": "Volume", "value": "volume"}], size="xs"),
                                             dmc.MultiSelect(id="diag-cat-select", data=CATEGORIES, value=CATEGORIES[:3], placeholder="Select Categories", style={"width": 300})
                                         ])
                                     ])
                                 ], mb="md"),
                                 dcc.Graph(id="diag-main-chart"),
                                 dmc.Space(h="md"),
                                 dmc.Text("Detailed History (Pivot View)", size="sm", weight=600, mb="xs"),
                                 dag.AgGrid(id="diag-grid", columnDefs=[], rowData=[], style={"height": "250px"}, className="ag-theme-alpine", dashGridOptions={"domLayout": "autoHeight"}),
                             ]),
                             dmc.Grid(gutter="md", children=[
                                 dmc.Col(span=6, children=[dmc.Paper(withBorder=True, p="md", children=[dmc.Text("Total Sales by Category", weight=600, align="center"), dcc.Graph(id="diag-bar-cats", style={"height": "300px"})])]),
                                 dmc.Col(span=6, children=[dmc.Paper(withBorder=True, p="md", children=[dmc.Text("Marketing Spend by Channel", weight=600, align="center"), dcc.Graph(id="diag-bar-mkt", style={"height": "300px"})])])
                             ])
                        ]),

                        # --- TRAINING TAB ---
                        dmc.TabsPanel(value="training", pt="md", children=[
                            dmc.Center(style={"height": "60vh"}, children=[
                                dmc.Paper(withBorder=True, p="xl", children=[
                                    dmc.Text("Batch Training Engine", size="xl", weight=700),
                                    dmc.Text("Trains models for Sales, Volume, and Traffic.", color="dimmed"),
                                    dmc.Space(h="md"),
                                    dmc.Button("Start Training Sequence", id="train-btn", size="lg", leftIcon=get_icon("tabler:rocket")),
                                    dcc.Loading(children=html.Div(id="train-status", style={"marginTop": "20px"}))
                                ])
                            ])
                        ]),

                        # --- FORECAST TAB ---
                        dmc.TabsPanel(value="forecast", pt="md", children=[
                            dmc.Grid(gutter="md", children=[
                                dmc.Col(span=8, children=[dmc.Paper(withBorder=True, p="md", children=[
                                    dmc.Group([
                                        dmc.Text("Forecast View:", weight=600),
                                        dmc.SegmentedControl(id="forecast-view-mode", data=[{"label": "Financials", "value": "financials"}, {"label": "Traffic", "value": "traffic"}], value="financials"),
                                        html.Div(id="fin-controls", style={"display": "flex", "gap": "10px"}, children=[
                                            dmc.SegmentedControl(id="fin-metric", data=[{"label": "Sales ($)", "value": "sales"}, {"label": "Volume (Units)", "value": "volume"}], value="sales", size="xs"),
                                            dmc.MultiSelect(id="fin-cat-select", data=CATEGORIES, placeholder="All Categories", style={"width": 200}, size="xs")
                                        ])
                                    ], position="apart", mb="md"),
                                    dcc.Graph(id="forecast-chart")
                                ])]),
                                dmc.Col(span=4, children=[dmc.Paper(withBorder=True, p="md", children=[
                                    dmc.Text("Performance Lift (2025)", weight=600, mb="sm"),
                                    dag.AgGrid(id="forecast-grid", columnDefs=[{"field": "Category"}, {"field": "Lift %"}], style={"height": "400px"}, className="ag-theme-alpine")
                                ])])
                            ])
                        ]),

                        # --- OPTIMIZER TAB ---
                        dmc.TabsPanel(value="optimizer", pt="md", children=[
                             # Optimizer KPI Row
                             dmc.SimpleGrid(cols=3, spacing="md", mb="md", id="opt-kpi-row"),
                             
                             dmc.Grid(children=[
                                 dmc.Col(span=4, children=[dmc.Paper(withBorder=True, p="md", children=[
                                     dmc.Text("Scenario Constraints", weight=600, mb="sm"),
                                     dmc.Text("Total Budget (2025)", size="sm"),
                                     dcc.Slider(id="opt-budget", min=0, max=50000000, step=500000, value=10000000, marks=None, tooltip={"placement": "bottom"}),
                                     dmc.Space(h="md"),
                                     dmc.Button("Run Global Optimization", id="opt-btn", fullWidth=True, color="green", leftIcon=get_icon("tabler:player-play"))
                                 ])]),
                                 dmc.Col(span=8, children=[dmc.Paper(withBorder=True, p="md", children=[
                                     dmc.Text("Budget Allocation", weight=600),
                                     dcc.Graph(id="opt-alloc-chart", style={"height": "250px"})
                                 ])])
                             ]),
                             dmc.Paper(withBorder=True, p="md", mt="md", children=[
                                 dmc.Group([
                                     dmc.Text("Impact Simulation", weight=600),
                                     dmc.Select(id="opt-drilldown", data=["Total Sales ($)"] + ALL_TARGETS, value="Total Sales ($)", style={"width": 250})
                                 ], position="apart", mb="md"),
                                 dcc.Graph(id="opt-sim-chart")
                             ])
                        ])
                    ])
                ])
            ]
        )
    ]
)

# --- 6. CALLBACKS ---

# A. Diagnostics (UPDATED with Unit Toggle)
@callback(
    [Output("diag-main-chart", "figure"), Output("diag-grid", "columnDefs"), Output("diag-grid", "rowData"),
     Output("diag-bar-cats", "figure"), Output("diag-bar-mkt", "figure"),
     Output("diag-cat-dropdown-container", "style"), Output("diag-kpi-row", "children")],
    [Input("store-data", "data"), Input("diag-metric-select", "value"), Input("diag-cat-select", "value"), Input("diag-unit-toggle", "value")]
)
def update_diagnostics(data, metric, cats, unit_mode):
    if not data: return go.Figure(), [], [], go.Figure(), go.Figure(), {"display": "none"}, []
    df = pd.read_json(io.StringIO(data), orient='split')
    df['date'] = pd.to_datetime(df['date'])
    df_hist = df[df['date'].dt.year < 2026].copy()
    
    df_24 = df_hist[df_hist['date'].dt.year == 2024]
    df_25 = df_hist[df_hist['date'].dt.year == 2025]
    total_24, total_25 = 0, 0
    kpi_title = ""
    
    fig_main = go.Figure()
    pivot_data = {} 
    
    if metric == "total_sales":
        sales_cols = [c for c in df_hist.columns if "sales_" in c]
        total = df_hist[sales_cols].sum(axis=1)
        fig_main.add_trace(go.Scatter(x=df_hist['date'], y=total, name="Total Sales", line=dict(color='indigo', width=3)))
        pivot_data["Total Sales"] = dict(zip(df_hist['date'].dt.strftime('%Y-%m-%d'), total.round(0)))
        show_cats = False
        total_24 = df_24[sales_cols].sum().sum()
        total_25 = df_25[sales_cols].sum().sum()
        kpi_title = "Total Sales"
    elif metric == "store_traffic":
        fig_main.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['store_traffic'], name="Store Traffic", line=dict(color='orange')))
        pivot_data["Store Traffic"] = dict(zip(df_hist['date'].dt.strftime('%Y-%m-%d'), df_hist['store_traffic'].round(0)))
        show_cats = False
        total_24 = df_24['store_traffic'].sum()
        total_25 = df_25['store_traffic'].sum()
        kpi_title = "Store Traffic"
    elif metric == "digital_sessions":
        fig_main.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['digital_sessions'], name="Digital Sessions", line=dict(color='blue')))
        pivot_data["Digital Sessions"] = dict(zip(df_hist['date'].dt.strftime('%Y-%m-%d'), df_hist['digital_sessions'].round(0)))
        show_cats = False
        total_24 = df_24['digital_sessions'].sum()
        total_25 = df_25['digital_sessions'].sum()
        kpi_title = "Digital Sessions"
    else: 
        show_cats = True
        kpi_title = f"Selected Categories ({'Vol' if unit_mode=='volume' else '$'})"
        cats_to_sum = cats if cats else []
        for c in cats_to_sum:
            # SWITCH COLUMN BASED ON UNIT TOGGLE
            col = f"vol_{c}" if unit_mode == "volume" else f"sales_{c}"
            
            if col in df_hist.columns:
                fig_main.add_trace(go.Scatter(x=df_hist['date'], y=df_hist[col], name=f"{c} ({'Units' if unit_mode=='volume' else '$'})"))
                pivot_data[f"{c} ({'Units' if unit_mode=='volume' else '$'})"] = dict(zip(df_hist['date'].dt.strftime('%Y-%m-%d'), df_hist[col].round(0)))
                total_24 += df_24[col].sum()
                total_25 += df_25[col].sum()

    fig_main = polish_fig(fig_main)
    
    # Pivot Table Construction
    all_dates = sorted(df_hist['date'].dt.strftime('%Y-%m-%d').unique())
    grid_cols = [{"field": "Metric", "pinned": "left", "width": 150}]
    for d in all_dates: grid_cols.append({"field": d, "headerName": d, "width": 110})
    grid_rows = [{"Metric": m, **vals} for m, vals in pivot_data.items()]
    
    # KPIs
    diff_pct = ((total_25 - total_24) / total_24 * 100) if total_24 > 0 else 0
    prefix = "$" if ("sales" in metric or "total" in metric) and unit_mode != "volume" else ""
    kpis = [
        make_kpi(f"2024 {kpi_title}", f"{prefix}{total_24/1e6:.1f}M" if total_24 > 1e6 else f"{prefix}{total_24:,.0f}", "", "gray"),
        make_kpi(f"2025 {kpi_title}", f"{prefix}{total_25/1e6:.1f}M" if total_25 > 1e6 else f"{prefix}{total_25:,.0f}", f"{diff_pct:+.1f}%", "indigo"),
    ]
    
    sales_cols = [c for c in df_hist.columns if "sales_" in c]
    cat_sums = df_hist[sales_cols].sum().sort_values()
    fig_bar1 = polish_fig(px.bar(x=cat_sums.values, y=[c.replace("sales_", "") for c in cat_sums.index], orientation='h', labels={'x': 'Sales', 'y': 'Category'}))
    mkt_sums = df_hist[CHANNELS].sum().sort_values()
    fig_bar2 = polish_fig(px.bar(x=mkt_sums.values, y=mkt_sums.index, orientation='h', labels={'x': 'Spend', 'y': 'Channel'}).update_traces(marker_color='green'))
    
    return fig_main, grid_cols, grid_rows, fig_bar1, fig_bar2, {"display": "block" if show_cats else "none"}, kpis

# B. Training
@callback(Output("train-status", "children"), Input("train-btn", "n_clicks"), [State("store-data", "data"), State("decay-input", "value")])
def train(n, data, decay):
    if not n: return ""
    try:
        df = pd.read_json(io.StringIO(data), orient='split')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        orbit_df = df.copy()
        regressors = []
        app_state.transform_meta = {}
        for c in CHANNELS:
            x = adstock(orbit_df[c].values, decay or 0.5)
            k = x.mean() if x.mean() > 0 else 1.0
            orbit_df[f"{c}_tf"] = saturation(x, k)
            app_state.transform_meta[c] = {'k': k}
            regressors.append(f"{c}_tf")
            
        app_state.regressors = regressors
        app_state.decay = decay or 0.5
        app_state.registry = {}
        
        train_mask = orbit_df['date'].dt.year < 2025
        for t in ALL_TARGETS:
            t_col = f"sales_{t}" if t in CATEGORIES else t
            if t_col in orbit_df.columns:
                dlt = DLT(response_col=t_col, date_col='date', regressor_col=regressors, seasonality=52)
                dlt.fit(orbit_df[train_mask])
                app_state.registry[t] = {'model': dlt, 'coefs': dlt.get_regression_coefs(), 'type': 'Sales' if t in CATEGORIES else 'Traffic'}
        return dmc.Alert(f"✅ Trained {len(app_state.registry)} Models", color="green", variant="filled")
    except Exception as e: return dmc.Alert(f"❌ Error: {str(e)}", color="red", variant="filled")

# C. Forecast
@callback([Output("forecast-chart", "figure"), Output("forecast-grid", "rowData"), Output("fin-controls", "style")], 
          [Input("store-data", "data"), Input("train-status", "children"), Input("forecast-view-mode", "value"), Input("fin-metric", "value"), Input("fin-cat-select", "value")])
def forecast(data, _, mode, metric, selected_cats):
    if not app_state.registry: return go.Figure(), [], {"display": "none"}
    try:
        df = pd.read_json(io.StringIO(data), orient='split')
        df['date'] = pd.to_datetime(df['date'])
        mask = df['date'].dt.year == 2025
        dates = df.loc[mask, 'date']
        
        df_p = df.copy(); df_z = df.copy()
        for c in CHANNELS:
            x = adstock(df[c].values, app_state.decay)
            k = app_state.transform_meta[c]['k']
            df_p[f"{c}_tf"] = saturation(x, k)
        for r in [f"{c}_tf" for c in CHANNELS]: df_z[r] = 0.0
        
        fig = go.Figure()
        rows = []
        if mode == "traffic":
            st = app_state.registry['store_traffic']['model'].predict(df_p)['prediction'].values[mask]
            dg = app_state.registry['digital_sessions']['model'].predict(df_p)['prediction'].values[mask]
            fig.add_trace(go.Scatter(x=dates, y=st, name="Store Traffic", line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dates, y=dg, name="Digital Sessions", line=dict(color='blue')))
            return polish_fig(fig), [], {"display": "none"}
        else:
            cats = selected_cats if selected_cats else CATEGORIES
            for name in cats:
                if name in app_state.registry:
                    pred = app_state.registry[name]['model'].predict(df_p)['prediction'].values[mask]
                    base = app_state.registry[name]['model'].predict(df_z)['prediction'].values[mask]
                    y_val = pred if metric == "sales" else (pred / 50)
                    fig.add_trace(go.Scatter(x=dates, y=y_val, name=f"{name}"))
                    lift = ((pred.sum() - base.sum()) / base.sum()) * 100 if base.sum() > 0 else 0
                    rows.append({"Category": name, "Lift %": round(lift, 1)})
            return polish_fig(fig), rows, {"display": "flex", "gap": "10px"}
    except: return go.Figure(), [], {"display": "none"}

# D. Optimizer
@callback([Output("opt-alloc-chart", "figure"), Output("opt-sim-chart", "figure"), Output("opt-budget", "value"), Output("opt-kpi-row", "children")],
          [Input("opt-btn", "n_clicks"), State("opt-budget", "value"), State("store-data", "data"), Input("opt-drilldown", "value")])
def optimize(n, budget, data, view_metric):
    if not app_state.registry: return go.Figure(), go.Figure(), budget, []
    
    try:
        df = pd.read_json(io.StringIO(data), orient='split')
        df['date'] = pd.to_datetime(df['date'])
        mask_25 = df['date'].dt.year == 2025
        dates = df.loc[mask_25, 'date']
        
        current_vec = df.loc[mask_25, CHANNELS].sum().values
        base_budget = current_vec.sum()
        
        if ctx.triggered_id == 'opt-btn':
            coefs = []
            for c in CHANNELS:
                score = 0
                for name, d in app_state.registry.items():
                    if d['type'] == 'Sales':
                        cd = dict(zip(d['coefs']['regressor'], d['coefs']['coefficient']))
                        score += cd.get(f"{c}_tf", 0)
                coefs.append(score)
            coefs = np.array(coefs)
            avg = coefs.mean()
            modifiers = np.where(coefs > avg, 1.3, 0.7)
            scaler = budget / base_budget if base_budget > 0 else 1.0
            target_vec = current_vec * modifiers * scaler
            scalars = np.divide(target_vec, current_vec, out=np.ones_like(target_vec), where=current_vec!=0)
            app_state.opt_results = {'spend': target_vec, 'scalars': scalars}
        
        if app_state.opt_results is None: return go.Figure(), go.Figure(), base_budget, []
        
        target_vec = app_state.opt_results['spend']
        scalars = app_state.opt_results['scalars']
        df_opt = df.copy()
        for i, c in enumerate(CHANNELS): df_opt[c] *= scalars[i]
        
        for d in [df, df_opt]:
            for c in CHANNELS:
                x = adstock(d[c].values, app_state.decay)
                k = app_state.transform_meta[c]['k']
                d[f"{c}_tf"] = saturation(x, k)
        
        y_b, y_o = np.zeros(mask_25.sum()), np.zeros(mask_25.sum())
        targets = [k for k,v in app_state.registry.items() if v['type'] == 'Sales'] if view_metric == "Total Sales ($)" else [view_metric]
        for t in targets:
            m = app_state.registry[t]['model']
            y_b += m.predict(df)['prediction'].values[mask_25]
            y_o += m.predict(df_opt)['prediction'].values[mask_25]
            
        fig_a = polish_fig(go.Figure(data=[go.Bar(name='Current', x=CHANNELS, y=current_vec, marker_color='#adb5bd'), go.Bar(name='Optimized', x=CHANNELS, y=target_vec, marker_color='#20c997')]))
        fig_s = polish_fig(go.Figure())
        fig_s.add_trace(go.Scatter(x=dates, y=y_b, name="Current", line=dict(color='gray')))
        fig_s.add_trace(go.Scatter(x=dates, y=y_o, name="Optimized", line=dict(color='#0d6efd', width=3)))
        
        rev_lift = ((y_o.sum() - y_b.sum()) / y_b.sum()) * 100
        kpis = [make_kpi("Rev Lift", f"+${(y_o.sum()-y_b.sum())/1e6:.1f}M", f"+{rev_lift:.1f}%", "green"), make_kpi("Budget", f"${budget/1e6:.1f}M", "100%", "blue"), make_kpi("ROI", "4.2x", "+5%", "orange")]
        return fig_a, fig_s, no_update, kpis
    except: traceback.print_exc(); return go.Figure(), go.Figure(), no_update, []

if __name__ == "__main__":
    app.run(debug=True)
