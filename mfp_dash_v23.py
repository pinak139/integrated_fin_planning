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
UNIT_ECONOMICS = {
    "Climbing": {"price": 120, "cost": 45, "inbound": 1.50, "outbound": 4.50, "cu_in": 600},
    "Snow":     {"price": 250, "cost": 110, "inbound": 3.00, "outbound": 8.00, "cu_in": 1200},
    "Running":  {"price": 130, "cost": 50,  "inbound": 1.20, "outbound": 4.00, "cu_in": 500},
    "Hiking":   {"price": 160, "cost": 65,  "inbound": 2.00, "outbound": 5.50, "cu_in": 800},
    "Yoga":     {"price": 80,  "cost": 20,  "inbound": 0.80, "outbound": 3.00, "cu_in": 300}
}
DEFAULT_ECONOMICS = {"price": 50, "cost": 15, "inbound": 0.50, "outbound": 2.00, "cu_in": 200}
DC_CAPACITY = {"DC West": {"Storage": 50e6, "Inbound": 2e6, "Outbound": 2.5e6}, "DC East": {"Storage": 80e6, "Inbound": 3.5e6, "Outbound": 4e6}}
PRICE_ELASTICITY = {"Climbing": 1.5, "Snow": 2.2, "Running": 1.8, "Hiking": 1.6, "Yoga": 2.5, "Accessories": 1.2}

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
    fig.update_layout(template="plotly_white", margin=dict(l=30, r=30, t=50, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified", legend=dict(orientation="h", y=1.1))
    if height: fig.update_layout(height=height)
    return fig

def make_kpi(title, value, diff, color="blue"):
    return dmc.Paper(children=[dmc.Group([dmc.Text(title, color="dimmed", size="xs", weight=700), dmc.ThemeIcon(DashIconify(icon="tabler:chart-bar", width=16), variant="light", color=color, radius="xl", size="sm")], position="apart", mb="xs"), dmc.Group([dmc.Text(value, weight=700, size="xl"), dmc.Badge(f"{diff}", color="green" if "+" in diff else "red", variant="light")], align="flex-end", spacing="xs")], withBorder=True, shadow="sm", p="md", radius="md")

def numpy_kmeans(data, k=3, max_iters=100):
    if len(data) < k: return np.zeros(len(data)), data
    n_samples, n_features = data.shape; centroids = data[np.random.choice(n_samples, k, replace=False)]
    for _ in range(max_iters):
        distances = np.sqrt(((data[:, np.newaxis] - centroids) ** 2).sum(axis=2)); labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i] for i in range(k)])
        if np.all(centroids == new_centroids): break
        centroids = new_centroids
    return labels, centroids

def adstock(x, a):
    y = np.zeros_like(x); y[0] = x[0]
    for t in range(1, len(x)): y[t] = x[t] + a * y[t-1]
    return y

def saturation(x, k): return x / (1 + (x/(k if k>0 else 1.0)))

# --- 3. DATA ENGINE ---
def generate_mock_data():
    dates = pd.date_range(start="2023-01-01", periods=156, freq="W-SUN"); cats = ['Climbing', 'Snow', 'Running', 'Hiking', 'Yoga', 'Accessories']; sales_data = {'date': dates}
    for c in cats: price = UNIT_ECONOMICS.get(c, DEFAULT_ECONOMICS)['price']; sales_data[f"sales_{c}"] = np.random.randint(50000, 150000, len(dates)); sales_data[f"vol_{c}"] = (sales_data[f"sales_{c}"] / price).astype(int)
    traffic_data = {'store_traffic': sales_data['sales_Hiking']/50, 'digital_sessions': sales_data['sales_Running']/10}; channels = ['Paid Search', 'Social', 'Display', 'Video']; mkt_data = {c: np.random.randint(5000, 20000, len(dates)) for c in channels}
    df = pd.DataFrame(sales_data); [df.__setitem__(k, v) for k,v in {**traffic_data, **mkt_data}.items()]; df['total_traffic'] = df['store_traffic'] + df['digital_sessions']
    return df, cats, channels

def generate_sc_data(df_main, categories):
    stores = []
    for i in range(1, 51): r = "West" if i <= 20 else "East"; base_temp = 65 if r == "West" else 45; stores.append({"Store ID": f"ST-{i:03d}", "Region": r, "DC Assignment": f"DC {r}", "Store Type": np.random.choice(["Flagship", "Mall"], p=[0.2, 0.8]), "Avg Temp (F)": int(np.random.normal(base_temp, 10)), "Vol_Multiplier": np.random.choice([1.5, 1.0, 0.6], p=[0.2, 0.5, 0.3])})
    df_stores = pd.DataFrame(stores); cols_lower = {c.lower(): c for c in df_main.columns}; total_sales_map = {s["Store ID"]: 0 for s in stores}
    for i, row in df_main.iterrows():
        for s_idx, store in df_stores.iterrows():
            store_share = store['Vol_Multiplier'] / df_stores['Vol_Multiplier'].sum()
            for cat in categories:
                eco = UNIT_ECONOMICS.get(cat, DEFAULT_ECONOMICS); real_col = cols_lower.get(f"vol_{cat}".lower())
                if real_col: total_sales_map[store['Store ID']] += row[real_col] * store_share * eco['price']
    df_stores['Total Revenue'] = df_stores['Store ID'].map(total_sales_map).astype(int)
    df_stores['Cluster'] = np.where(df_stores['Vol_Multiplier'] > 1.2, "0", np.where(df_stores['Vol_Multiplier'] < 0.8, "2", "1"))
    
    sc_rows_dc = []
    for i, row in df_main.iterrows():
        date = row['date']
        for dc in ["DC West", "DC East"]:
            dc_share = 0.4 if dc == "DC West" else 0.6
            for cat in categories:
                eco = UNIT_ECONOMICS.get(cat, DEFAULT_ECONOMICS); real_col = cols_lower.get(f"vol_{cat}".lower())
                if not real_col: continue
                dem = row[real_col]; out = int(dem * dc_share); inp = int(out * np.random.normal(1.02, 0.05))
                sc_rows_dc.append({"date": date, "DC": dc, "Category": cat, "Units_Out": out, "Units_In": inp, "CuIn_Out": out * eco['cu_in'], "CuIn_In": inp * eco['cu_in'], "Demand_Dollars": out * eco['price'], "Margin": (out * eco['price']) - (out * eco['cost'] + inp * eco['inbound'] + out * eco['outbound']), "Cost_Material": out * eco['cost'], "Cost_Inbound": inp * eco['inbound'], "Cost_Outbound": out * eco['outbound']})
    df_sc = pd.DataFrame(sc_rows_dc)
    if not df_sc.empty:
        df_agg = df_sc.groupby(['date', 'DC']).agg({'CuIn_In': 'sum', 'CuIn_Out': 'sum'}).reset_index().sort_values('date'); curr = {"DC West": 30e6, "DC East": 50e6}; invs = []
        for i, r in df_agg.iterrows(): dc = r['DC']; curr[dc] = max(0, curr[dc] + (r['CuIn_In'] - r['CuIn_Out'])); invs.append(curr[dc])
        df_agg['CuIn_Inventory'] = invs; df_sc = pd.merge(df_sc, df_agg[['date', 'DC', 'CuIn_Inventory']], on=['date', 'DC'], how='left')
    return df_stores, df_sc

def generate_customer_data(df_stores, categories):
    n_cust = 1000; cust_ids = [f"CUST-{i:04d}" for i in range(n_cust)]; store_probs = df_stores['Vol_Multiplier'] / df_stores['Vol_Multiplier'].sum(); assigned_stores = np.random.choice(df_stores['Store ID'], n_cust, p=store_probs); segments = ["VIP", "High Value", "Mid", "Low"]; assigned_segments = np.random.choice(segments, n_cust, p=[0.05, 0.15, 0.40, 0.40])
    df_cust = pd.DataFrame({"Customer ID": cust_ids, "Primary Store": assigned_stores, "Segment": assigned_segments, "CLV": np.where(assigned_segments=="VIP", np.random.randint(2000,5000, n_cust), np.random.randint(100,800, n_cust))})
    for cat in categories: base_prop = np.random.beta(2, 5, n_cust); base_prop = np.where(assigned_segments == "VIP", base_prop * 1.5, base_prop); df_cust[f"Propensity_{cat}"] = np.clip(base_prop, 0, 1)
    return df_cust

def generate_transaction_history(df_cust, categories):
    history = []
    for idx, row in df_cust.iterrows():
        n_trans = np.random.randint(5, 12); cust_id = row['Customer ID']; cat_probs = [row[f"Propensity_{c}"] for c in categories]; cat_probs = np.array(cat_probs) / sum(cat_probs)
        for _ in range(n_trans):
            cat = np.random.choice(categories, p=cat_probs); eco = UNIT_ECONOMICS.get(cat, DEFAULT_ECONOMICS); price = eco['price'] * np.random.normal(1, 0.1); date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            history.append({"Customer ID": cust_id, "Date": date.strftime('%Y-%m-%d'), "Category": cat, "Product": f"{cat} Item {np.random.randint(100,999)}", "Amount": round(price, 2)})
    return pd.DataFrame(history)

def generate_assortment_data(df_main, categories):
    cols_lower = {c.lower(): c for c in df_main.columns}; cat_volumes = {}; df_2025 = df_main[df_main['date'].dt.year == 2025]
    for cat in categories: target_col = f"vol_{cat}".lower(); real_col = cols_lower.get(target_col); cat_volumes[cat] = df_2025[real_col].mean() if real_col else 1000
    styles = []
    for cat in categories:
        num_styles = np.random.randint(15, 25); eco = UNIT_ECONOMICS.get(cat, DEFAULT_ECONOMICS); target_vol = cat_volumes.get(cat, 1000); avg_style_vol = (target_vol / num_styles) * 1.2
        for i in range(num_styles):
            style_id = f"{cat[:3].upper()}-{np.random.randint(100, 999)}"; price = eco['price'] * np.random.normal(1, 0.1); cost = eco['cost'] * np.random.normal(1, 0.05); vol_potential = int(avg_style_vol * np.random.normal(1, 0.4)); vol_potential = max(5, vol_potential)
            styles.append({"Style ID": style_id, "Name": f"{cat} {np.random.choice(['Pro', 'Elite', 'Basic', 'Lite', 'Max'])} {i+1}", "Category": cat, "Cost": round(cost, 2), "Price": round(price, 2), "Margin $": round(price - cost, 2), "Base Weekly Units": vol_potential, "Linear Ft": 0.5, "Margin Efficiency ($/Ft)": round(((price - cost) * vol_potential) / 0.5, 2)})
    return pd.DataFrame(styles)

def generate_item_lifecycle(df_assort):
    items = []
    base_curve = np.array([0.05, 0.10, 0.15, 0.20, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01])
    top_items = df_assort.sort_values('Base Weekly Units', ascending=False).head(20)
    for _, row in top_items.iterrows():
        total_buy_qty = row['Base Weekly Units'] * 13; current_week = 6; plan = (base_curve * total_buy_qty).astype(int); perf_factor = np.random.choice([0.8, 1.0, 1.2]); actuals = (plan[:current_week] * perf_factor * np.random.normal(1, 0.1, current_week)).astype(int)
        items.append({"Item": row['Name'], "Category": row['Category'], "Total Buy": total_buy_qty, "Performance": perf_factor, "Current Week": current_week, "SOH": total_buy_qty - actuals.sum(), "Plan_Curve": plan.tolist(), "Actual_Curve": actuals.tolist()})
    return pd.DataFrame(items)

def load_data(): return generate_mock_data()

# --- 4. APP INIT ---
df_main, CATEGORIES, CHANNELS = load_data()
df_stores, df_sc_data = generate_sc_data(df_main, CATEGORIES)
df_cust = generate_customer_data(df_stores, CATEGORIES)
df_transactions = generate_transaction_history(df_cust, CATEGORIES)
df_assortment = generate_assortment_data(df_main, CATEGORIES)
df_lifecycle = generate_item_lifecycle(df_assortment)
ALL_TARGETS = CATEGORIES + ['store_traffic', 'digital_sessions']
app = dash.Dash(__name__)

# --- 5. LAYOUT ---
app.layout = dmc.MantineProvider(
    theme={"fontFamily": "'Inter', sans-serif", "primaryColor": "indigo"},
    children=[
        dcc.Store(id="store-data", data=df_main.to_json(date_format='iso', orient='split')),
        dcc.Store(id="sc-data", data=df_sc_data.to_json(date_format='iso', orient='split')),
        dcc.Store(id="stores-meta", data=df_stores.to_json(date_format='iso', orient='split')),
        dcc.Store(id="cust-data", data=df_cust.to_json(date_format='iso', orient='split')),
        dcc.Store(id="trans-data", data=df_transactions.to_json(date_format='iso', orient='split')),
        dcc.Store(id="assort-data", data=df_assortment.to_json(date_format='iso', orient='split')),
        dcc.Store(id="lifecycle-data", data=df_lifecycle.to_json(date_format='iso', orient='split')),
        dcc.Store(id="clustered-store-data"), 
        
        dmc.AppShell(
            header=dmc.Header(height=60, p="md", children=[dmc.Group(position="apart", children=[dmc.Group([DashIconify(icon="tabler:mountain", color="#4c6ef5", width=30), dmc.Text("PEAK 28 | Planning", size="lg", weight=700)]), dmc.Badge("v13.0 Final Master", color="green", variant="filled")])]),
            navbar=dmc.Navbar(width={"base": 300}, p="md", children=[dmc.Text("Settings", size="sm", weight=500, color="dimmed", mb="md"), dmc.NumberInput(label="Growth Target", id="growth-input", value=0.10, step=0.01, precision=2, mb="sm"), dmc.NumberInput(label="Adstock Decay", id="decay-input", value=0.5, step=0.1, min=0, max=0.9, mb="md"), dmc.Divider(my="lg"), dmc.NavLink(label="System Status", icon=get_icon("tabler:server"), variant="subtle", active=True, color="gray")]),
            children=[
                dmc.Container(fluid=True, children=[
                    dmc.Tabs(value="executive", variant="outline", children=[
                        dmc.TabsList([
                            dmc.Tab("Executive", value="executive", icon=get_icon("tabler:activity-heartbeat")), # NEW LANDING
                            dmc.Tab("Diagnostics", value="diagnostics", icon=get_icon("tabler:dashboard")),
                            dmc.Tab("Training", value="training", icon=get_icon("tabler:brain")),
                            dmc.Tab("Forecast", value="forecast", icon=get_icon("tabler:chart-arrows")),
                            dmc.Tab("Supply Chain", value="supply_chain", icon=get_icon("tabler:truck")),
                            dmc.Tab("Store Clusters", value="clusters", icon=get_icon("tabler:circles-relation")),
                            dmc.Tab("Assortment", value="assortment", icon=get_icon("tabler:hanger")),
                            dmc.Tab("Item Plan", value="item_plan", icon=get_icon("tabler:shirt")),
                            dmc.Tab("Customer 360", value="customer360", icon=get_icon("tabler:user-search")),
                            dmc.Tab("Optimizer", value="optimizer", icon=get_icon("tabler:adjustments")),
                        ]),
                        
                        # --- EXECUTIVE SUMMARY (NEW) ---
                        dmc.TabsPanel(value="executive", pt="md", children=[
                            dmc.SimpleGrid(cols=4, spacing="md", mb="lg", id="exec-kpi-row"),
                            dmc.Grid(gutter="md", children=[
                                dmc.Col(span=8, children=[
                                    dmc.Paper(withBorder=True, p="md", mb="md", children=[
                                        dmc.Text("Strategic Pulse: Revenue vs Inventory", weight=700, size="lg", mb="sm"),
                                        dcc.Graph(id="exec-pulse-chart")
                                    ])
                                ]),
                                dmc.Col(span=4, children=[
                                    dmc.Paper(withBorder=True, p="md", children=[
                                        dmc.Text("Profitability by Category", weight=700, size="lg", mb="sm"),
                                        dcc.Graph(id="exec-tree-chart", style={"height": "350px"})
                                    ])
                                ])
                            ])
                        ]),

                        # (OTHER TABS AS BEFORE)
                        dmc.TabsPanel(value="diagnostics", pt="md", children=[dmc.SimpleGrid(cols=3, spacing="md", mb="md", id="diag-kpi-row"), dmc.Paper(withBorder=True, p="md", mb="md", children=[dmc.Group([dmc.Text("Metric View:", weight=600), dmc.SegmentedControl(id="diag-metric-select", value="total_sales", data=[{"label": "Total Sales", "value": "total_sales"}, {"label": "Traffic", "value": "store_traffic"}, {"label": "Category", "value": "category"}]), html.Div(id="diag-cat-dropdown-container", style={"display": "none"}, children=[dmc.Group([dmc.SegmentedControl(id="diag-unit-toggle", value="sales", data=[{"label": "$", "value": "sales"}, {"label": "Vol", "value": "volume"}], size="xs"), dmc.MultiSelect(id="diag-cat-select", data=CATEGORIES, value=CATEGORIES[:3], placeholder="Cats", style={"width": 300})])])], position="apart", mb="md"), dcc.Graph(id="diag-main-chart"), dag.AgGrid(id="diag-grid", columnDefs=[], rowData=[], style={"height": "250px"}, className="ag-theme-alpine", dashGridOptions={"domLayout": "autoHeight"})]), dmc.Grid(gutter="md", children=[dmc.Col(span=6, children=[dmc.Paper(withBorder=True, p="md", children=[dcc.Graph(id="diag-bar-cats", style={"height": "300px"})])]), dmc.Col(span=6, children=[dmc.Paper(withBorder=True, p="md", children=[dcc.Graph(id="diag-bar-mkt", style={"height": "300px"})])])])]),
                        dmc.TabsPanel(value="training", pt="md", children=[dmc.Center(style={"height": "60vh"}, children=[dmc.Paper(withBorder=True, p="xl", children=[dmc.Text("Batch Training Engine", size="xl", weight=700), dmc.Button("Start Training Sequence", id="train-btn", size="lg", leftIcon=get_icon("tabler:rocket")), dcc.Loading(children=html.Div(id="train-status", style={"marginTop": "20px"}))])])]),
                        dmc.TabsPanel(value="forecast", pt="md", children=[dmc.Grid(gutter="md", children=[dmc.Col(span=8, children=[dmc.Paper(withBorder=True, p="md", children=[dmc.Group([dmc.Text("Forecast View:", weight=600), dmc.SegmentedControl(id="forecast-view-mode", data=[{"label": "Financials", "value": "financials"}, {"label": "Traffic", "value": "traffic"}], value="financials"), html.Div(id="fin-controls", style={"display": "flex", "gap": "10px"}, children=[dmc.SegmentedControl(id="fin-metric", data=[{"label": "Sales ($)", "value": "sales"}, {"label": "Volume (Units)", "value": "volume"}], value="sales", size="xs"), dmc.MultiSelect(id="fin-cat-select", data=CATEGORIES, placeholder="All Categories", style={"width": 200}, size="xs")])], position="apart", mb="md"), dcc.Graph(id="forecast-chart")])]), dmc.Col(span=4, children=[dmc.Paper(withBorder=True, p="md", children=[dmc.Text("Performance Lift (2025)", weight=600, mb="sm"), dag.AgGrid(id="forecast-grid", columnDefs=[{"field": "Category"}, {"field": "Lift %"}], style={"height": "400px"}, className="ag-theme-alpine")])])])]),
                        dmc.TabsPanel(value="supply_chain", pt="md", children=[dmc.Grid(gutter="md", children=[dmc.Col(span=3, children=[dmc.Paper(withBorder=True, p="md", mb="md", children=[dmc.Text("View Mode", weight=600, mb="sm"), dmc.SegmentedControl(id="sc-view-mode", value="financials", data=[{"label": "Financials", "value": "financials"}, {"label": "Capacity & Vol", "value": "capacity"}], fullWidth=True, mb="md"), dmc.Select(id="sc-dc-select", label="Location", data=["Total Network", "DC West", "DC East"], value="Total Network", mb="md"), dmc.Select(id="sc-cat-select", label="Category", data=["All Categories"] + CATEGORIES, value="All Categories", mb="xl"), dmc.Text("Network Map", weight=600, mb="sm"), dag.AgGrid(id="sc-store-grid", columnDefs=[{"field": "Store ID"}, {"field": "DC Assignment"}], rowData=df_stores.to_dict("records"), style={"height": "300px"}, className="ag-theme-alpine")])]), dmc.Col(span=9, children=[dmc.Paper(withBorder=True, p="md", mb="md", children=[dmc.Group([html.Div(id="sc-chart-title", children=dmc.Text("Supply Chain Analysis", weight=700, size="lg")), dmc.Badge("2025 Plan", color="blue")], position="apart", mb="md"), dcc.Graph(id="sc-main-chart", style={"height": "400px"}), html.Div(id="sc-secondary-chart-container", children=[dmc.Divider(label="Flow Constraints", labelPosition="center", my="lg"), dcc.Graph(id="sc-flow-chart", style={"height": "300px"})], style={"display": "none"})]), dmc.Paper(withBorder=True, p="md", children=[dmc.Text("Weekly Detail", weight=600, mb="sm"), dag.AgGrid(id="sc-detail-grid", columnDefs=[], rowData=[], style={"height": "250px"}, className="ag-theme-alpine")])])])]),
                        dmc.TabsPanel(value="clusters", pt="md", children=[dmc.Grid(gutter="md", children=[dmc.Col(span=3, children=[dmc.Paper(withBorder=True, p="md", children=[dmc.Text("Clustering Parameters", weight=600, mb="sm"), dmc.Text("Number of Clusters (k)", size="sm", mb="xs"), dcc.Slider(id="cluster-k-slider", min=2, max=6, step=1, value=3, marks={i: str(i) for i in range(2, 7)})])]), dmc.Col(span=9, children=[dmc.Paper(withBorder=True, p="md", mb="md", children=[dmc.Text("Store Performance vs. Environment", weight=700, size="lg", mb="md"), dcc.Graph(id="cluster-scatter")]), dmc.Paper(withBorder=True, p="md", children=[dmc.Text("Cluster Profiles", weight=600, mb="sm"), dag.AgGrid(id="cluster-grid", columnDefs=[{"field": "Store ID"}, {"field": "Cluster"}, {"field": "Region"}, {"field": "Total Revenue", "valueFormatter": {"function": "d3.format('$,.0f')(params.value)"}}, {"field": "Avg Temp (F)"}], rowData=[], className="ag-theme-alpine", style={"height": "300px"})])])])]),
                        dmc.TabsPanel(value="assortment", pt="md", children=[dmc.Grid(gutter="md", children=[dmc.Col(span=3, children=[dmc.Paper(withBorder=True, p="md", children=[dmc.Text("Assortment Plan", weight=600, mb="md"), dmc.Select(id="as-cluster-select", label="Planning Cluster", data=[], value=None, placeholder="Select a Cluster", mb="sm"), dmc.Select(id="as-cat-select", label="Category", data=CATEGORIES, value=CATEGORIES[0], mb="md"), dmc.Text("Shelf Space (Linear Ft)", size="sm", mb="xs"), dcc.Slider(id="as-space-slider", min=10, max=100, step=5, value=50, marks={10: '10', 50: '50', 100: '100'}), dmc.Space(h="md"), dmc.Button("Optimize Assortment", id="as-opt-btn", fullWidth=True, color="indigo", leftIcon=get_icon("tabler:wand")), dmc.Divider(my="lg"), dmc.Text("Results", weight=600, mb="sm"), html.Div(id="as-kpi-container")])]), dmc.Col(span=9, children=[dmc.Paper(withBorder=True, p="md", mb="md", children=[dmc.Text("Customer Profile (CLV & Propensity)", weight=700, size="lg", mb="sm"), dcc.Graph(id="as-cust-profile-chart", style={"height": "250px"})]), dmc.Paper(withBorder=True, p="md", mb="md", children=[dmc.Group([dmc.Text("Assortment Efficiency Frontier", weight=700, size="lg"), html.Div(id="as-cluster-badge")], position="apart"), dcc.Graph(id="as-scatter-chart")]), dmc.Paper(withBorder=True, p="md", children=[dmc.Text("Optimized Planogram List", weight=600, mb="sm"), dag.AgGrid(id="as-product-grid", columnDefs=[{"field": "Status", "cellStyle": {"styleConditions": [{"condition": "params.value == 'Keep'", "style": {"backgroundColor": "#d1fae5", "color": "#065f46"}}, {"condition": "params.value == 'Drop'", "style": {"backgroundColor": "#fee2e2", "color": "#991b1b"}}]}}, {"field": "Name"}, {"field": "Price"}, {"field": "Propensity Lift", "headerName": "Cust. Lift %", "cellStyle": {'color': 'blue'}}, {"field": "Adj. Margin Efficiency ($/Ft)"}], rowData=[], style={"height": "400px"}, className="ag-theme-alpine")])])])]),
                        dmc.TabsPanel(value="item_plan", pt="md", children=[dmc.Grid(gutter="md", children=[dmc.Col(span=4, children=[dmc.Paper(withBorder=True, p="md", children=[dmc.Text("In-Season Control", weight=600, mb="md"), dmc.Select(id="item-select", label="Select Item", data=[], searchable=True, mb="lg"), dmc.Divider(label="Markdown Scenario", labelPosition="center", my="lg"), dmc.Text("Discount %", size="sm"), dcc.Slider(id="item-markdown", min=0, max=0.7, step=0.05, value=0, marks={0:'0%', 0.25:'25%', 0.5:'50%', 0.7:'70%'}), dmc.Space(h="md"), html.Div(id="item-kpi-card")])]), dmc.Col(span=8, children=[dmc.Paper(withBorder=True, p="md", mb="md", children=[dmc.Group([dmc.Text("Weekly Sales Curve (Plan vs Actual)", weight=700), dmc.Badge("Connected to Assortment", color="indigo")], position="apart"), dcc.Graph(id="item-curve-chart")]), dmc.Paper(withBorder=True, p="md", children=[dmc.Text("Cumulative Sell-Through", weight=600, mb="sm"), dcc.Graph(id="item-cum-chart", style={"height": "250px"})])])])]),
                        dmc.TabsPanel(value="customer360", pt="md", children=[dmc.Grid(gutter="md", children=[dmc.Col(span=4, children=[dmc.Paper(withBorder=True, p="md", children=[dmc.Text("Customer Inspector", weight=600, mb="md"), dmc.Select(id="c360-select", label="Search Customer ID", placeholder="Select ID...", data=[f"CUST-{i:04d}" for i in range(100)], value="CUST-0001", searchable=True, mb="lg"), html.Div(id="c360-profile-card")])]), dmc.Col(span=8, children=[dmc.Paper(withBorder=True, p="md", mb="md", children=[dmc.Text("Propensity Profile vs Network Avg", weight=700, size="lg"), dcc.Graph(id="c360-radar-chart", style={"height": "300px"})]), dmc.Paper(withBorder=True, p="md", children=[dmc.Text("Transaction History", weight=600, mb="sm"), dag.AgGrid(id="c360-trans-grid", columnDefs=[{"field": "Date"}, {"field": "Category"}, {"field": "Product"}, {"field": "Amount"}], rowData=[], className="ag-theme-alpine", style={"height": "250px"})])])])]),
                        dmc.TabsPanel(value="optimizer", pt="md", children=[dmc.SimpleGrid(cols=3, spacing="md", mb="md", id="opt-kpi-row"), dmc.Grid(children=[dmc.Col(span=4, children=[dmc.Paper(withBorder=True, p="md", children=[dmc.Text("Scenario Constraints", weight=600, mb="sm"), dmc.Text("Total Budget (2025)", size="sm"), dcc.Slider(id="opt-budget", min=0, max=50000000, step=500000, value=10000000, marks=None, tooltip={"placement": "bottom"}), dmc.Space(h="md"), dmc.Button("Run Global Optimization", id="opt-btn", fullWidth=True, color="green", leftIcon=get_icon("tabler:player-play"))])]), dmc.Col(span=8, children=[dmc.Paper(withBorder=True, p="md", children=[dmc.Text("Budget Allocation", weight=600), dcc.Graph(id="opt-alloc-chart", style={"height": "250px"})])])]), dmc.Paper(withBorder=True, p="md", mt="md", children=[dmc.Group([dmc.Text("Impact Simulation", weight=600), dmc.Select(id="opt-drilldown", data=["Total Sales ($)"] + ALL_TARGETS, value="Total Sales ($)", style={"width": 250})], position="apart", mb="md"), dcc.Graph(id="opt-sim-chart")])])
                    ])
                ])
            ]
        )
    ]
)

# --- 6. CALLBACKS ---

# NEW: EXECUTIVE SUMMARY
@callback(
    [Output("exec-kpi-row", "children"), Output("exec-pulse-chart", "figure"), Output("exec-tree-chart", "figure")],
    [Input("store-data", "data"), Input("sc-data", "data"), Input("assort-data", "data")]
)
def update_executive_dashboard(store_json, sc_json, assort_json):
    if not store_json or not sc_json: return [], go.Figure(), go.Figure()
    
    # Load Data
    df_sales = pd.read_json(io.StringIO(store_json), orient='split'); df_sales['date'] = pd.to_datetime(df_sales['date'])
    df_sc = pd.read_json(io.StringIO(sc_json), orient='split'); df_sc['date'] = pd.to_datetime(df_sc['date'])
    
    # 2025 Filter
    df_sales_25 = df_sales[df_sales['date'].dt.year == 2025]
    df_sc_25 = df_sc[df_sc['date'].dt.year == 2025]
    
    # 1. KPIs
    # Calculate Total Revenue
    sales_cols = [c for c in df_sales.columns if "sales_" in c]
    total_rev = df_sales_25[sales_cols].sum().sum()
    
    # Calculate Margin (Weighted Est)
    total_margin = df_sc_25['Margin'].sum()
    margin_pct = (total_margin / total_rev * 100) if total_rev > 0 else 0
    
    # Inventory Health
    avg_inv = df_sc_25['CuIn_Inventory'].mean()
    turns = (total_rev / 50) / (avg_inv / 1000) # Proxy calculation
    
    kpis = [
        make_kpi("2025 Revenue Forecast", f"${total_rev/1e6:.1f}M", "+12% vs LY", "green"),
        make_kpi("Projected Net Margin", f"${total_margin/1e6:.1f}M", f"{margin_pct:.1f}%", "indigo"),
        make_kpi("Inventory Turnover", f"{turns:.1f}x", "Target: 4.0x", "blue"),
        make_kpi("Service Level Risk", "Low", "98.5%", "green")
    ]
    
    # 2. Pulse Chart (Dual Axis)
    # Aggregating weekly
    df_pulse = df_sc_25.groupby("date").agg({"Demand_Dollars": "sum", "CuIn_Inventory": "mean"}).reset_index()
    
    fig_pulse = go.Figure()
    fig_pulse.add_trace(go.Bar(x=df_pulse['date'], y=df_pulse['Demand_Dollars'], name="Revenue ($)", marker_color='indigo', opacity=0.6))
    fig_pulse.add_trace(go.Scatter(x=df_pulse['date'], y=df_pulse['CuIn_Inventory'], name="Inventory Level", yaxis='y2', line=dict(color='orange', width=3)))
    fig_pulse = polish_fig(fig_pulse)
    fig_pulse.update_layout(title="Enterprise Pulse: Demand vs. Stock", yaxis2=dict(overlaying='y', side='right', title="Inventory Vol"))
    
    # 3. Treemap
    # Aggregating by Category
    df_tree = df_sc_25.groupby("Category").agg({"Demand_Dollars": "sum", "Margin": "sum"}).reset_index()
    df_tree['Margin %'] = df_tree['Margin'] / df_tree['Demand_Dollars']
    
    fig_tree = px.treemap(df_tree, path=['Category'], values='Demand_Dollars', color='Margin %', color_continuous_scale='RdBu', title="Category Performance Matrix")
    fig_tree = polish_fig(fig_tree)
    
    return kpis, fig_pulse, fig_tree

# (EXISTING CALLBACKS PRESERVED BELOW)
@callback([Output("diag-main-chart", "figure"), Output("diag-grid", "columnDefs"), Output("diag-grid", "rowData"), Output("diag-bar-cats", "figure"), Output("diag-bar-mkt", "figure"), Output("diag-cat-dropdown-container", "style"), Output("diag-kpi-row", "children")], [Input("store-data", "data"), Input("diag-metric-select", "value"), Input("diag-cat-select", "value"), Input("diag-unit-toggle", "value")])
def update_diagnostics(data, metric, cats, unit_mode):
    if not data: return go.Figure(), [], [], go.Figure(), go.Figure(), {"display": "none"}, []
    df = pd.read_json(io.StringIO(data), orient='split'); df['date'] = pd.to_datetime(df['date']); df_hist = df[df['date'].dt.year < 2026].copy(); cols_lower = {c.lower(): c for c in df_hist.columns}; df_24 = df_hist[df_hist['date'].dt.year == 2024]; df_25 = df_hist[df_hist['date'].dt.year == 2025]; total_24, total_25 = 0, 0; kpi_title = ""; fig_main = go.Figure(); pivot_data = {}
    if metric == "total_sales": sales_cols = [c for c in df_hist.columns if "sales_" in c]; total = df_hist[sales_cols].sum(axis=1); fig_main.add_trace(go.Scatter(x=df_hist['date'], y=total, name="Total Sales", line=dict(color='indigo', width=3))); pivot_data["Total Sales"] = dict(zip(df_hist['date'].dt.strftime('%Y-%m-%d'), total.round(0))); show_cats = False; total_24 = df_24[sales_cols].sum().sum(); total_25 = df_25[sales_cols].sum().sum(); kpi_title = "Total Sales"
    elif metric == "store_traffic": fig_main.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['store_traffic'], name="Store Traffic", line=dict(color='orange'))); pivot_data["Store Traffic"] = dict(zip(df_hist['date'].dt.strftime('%Y-%m-%d'), df_hist['store_traffic'].round(0))); show_cats = False; total_24 = df_24['store_traffic'].sum(); total_25 = df_25['store_traffic'].sum(); kpi_title = "Store Traffic"
    elif metric == "category": show_cats = True; kpi_title = f"Selected ({'Vol' if unit_mode=='volume' else '$'})"; cats_to_sum = cats if cats else []
    for c in (cats if metric == "category" else []):
        prefix = "vol_" if unit_mode == "volume" else "sales_"; target_col = f"{prefix}{c}".lower(); real_col = cols_lower.get(target_col)
        if real_col: fig_main.add_trace(go.Scatter(x=df_hist['date'], y=df_hist[real_col], name=f"{c}")); pivot_data[f"{c}"] = dict(zip(df_hist['date'].dt.strftime('%Y-%m-%d'), df_hist[real_col].round(0))); total_24 += df_24[real_col].sum(); total_25 += df_25[real_col].sum()
    fig_main = polish_fig(fig_main); all_dates = sorted(df_hist['date'].dt.strftime('%Y-%m-%d').unique()); grid_cols = [{"field": "Metric", "pinned": "left", "width": 150}]; [grid_cols.append({"field": d, "headerName": d, "width": 110}) for d in all_dates]; grid_rows = [{"Metric": m, **vals} for m, vals in pivot_data.items()]; diff_pct = ((total_25 - total_24) / total_24 * 100) if total_24 > 0 else 0; prefix = "$" if ("sales" in metric or "total" in metric) and unit_mode != "volume" else ""; kpis = [make_kpi(f"2024 {kpi_title}", f"{prefix}{total_24/1e6:.1f}M", "", "gray"), make_kpi(f"2025 {kpi_title}", f"{prefix}{total_25/1e6:.1f}M", f"{diff_pct:+.1f}%", "indigo")]; sales_cols = [c for c in df_hist.columns if "sales_" in c]; cat_sums = df_hist[sales_cols].sum().sort_values(); fig_bar1 = polish_fig(px.bar(x=cat_sums.values, y=[c.replace("sales_", "") for c in cat_sums.index], orientation='h', labels={'x': 'Sales', 'y': 'Category'})); mkt_sums = df_hist[CHANNELS].sum().sort_values(); fig_bar2 = polish_fig(px.bar(x=mkt_sums.values, y=mkt_sums.index, orientation='h', labels={'x': 'Spend', 'y': 'Channel'}).update_traces(marker_color='green'))
    return fig_main, grid_cols, grid_rows, fig_bar1, fig_bar2, {"display": "block" if show_cats else "none"}, kpis

@callback(Output("train-status", "children"), Input("train-btn", "n_clicks"), [State("store-data", "data"), State("decay-input", "value")])
def train(n, data, decay):
    if not n: return ""
    try:
        df = pd.read_json(io.StringIO(data), orient='split'); df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date').reset_index(drop=True); orbit_df = df.copy(); regressors = []; app_state.transform_meta = {}
        for c in CHANNELS: x = adstock(orbit_df[c].values, decay or 0.5); k = x.mean() if x.mean() > 0 else 1.0; orbit_df[f"{c}_tf"] = saturation(x, k); app_state.transform_meta[c] = {'k': k}; regressors.append(f"{c}_tf")
        app_state.regressors = regressors; app_state.decay = decay or 0.5; app_state.registry = {}
        train_mask = orbit_df['date'].dt.year < 2025
        for t in ALL_TARGETS:
            t_col = t
            if t in CATEGORIES: sales_cols = {c.lower(): c for c in orbit_df.columns if 'sales_' in c}; t_col = sales_cols.get(f"sales_{t}".lower())
            if t_col and t_col in orbit_df.columns: dlt = DLT(response_col=t_col, date_col='date', regressor_col=regressors, seasonality=52); dlt.fit(orbit_df[train_mask]); app_state.registry[t] = {'model': dlt, 'coefs': dlt.get_regression_coefs(), 'type': 'Sales' if t in CATEGORIES else 'Traffic'}
        return dmc.Alert(f"✅ Trained {len(app_state.registry)} Models", color="green", variant="filled")
    except Exception as e: return dmc.Alert(f"❌ Error: {str(e)}", color="red", variant="filled")

@callback([Output("forecast-chart", "figure"), Output("forecast-grid", "rowData"), Output("fin-controls", "style")], [Input("store-data", "data"), Input("train-status", "children"), Input("forecast-view-mode", "value"), Input("fin-metric", "value"), Input("fin-cat-select", "value")])
def forecast(data, _, mode, metric, selected_cats):
    if not app_state.registry: return go.Figure(), [], {"display": "none"}
    try:
        df = pd.read_json(io.StringIO(data), orient='split'); df['date'] = pd.to_datetime(df['date']); mask = df['date'].dt.year == 2025; dates = df.loc[mask, 'date']; df_p = df.copy(); df_z = df.copy()
        for c in CHANNELS: x = adstock(df[c].values, app_state.decay); k = app_state.transform_meta[c]['k']; df_p[f"{c}_tf"] = saturation(x, k)
        for r in [f"{c}_tf" for c in CHANNELS]: df_z[r] = 0.0
        fig = go.Figure(); rows = []
        if mode == "traffic":
            st = app_state.registry['store_traffic']['model'].predict(df_p)['prediction'].values[mask]; dg = app_state.registry['digital_sessions']['model'].predict(df_p)['prediction'].values[mask]; fig.add_trace(go.Scatter(x=dates, y=st, name="Store Traffic", line=dict(color='orange'))); fig.add_trace(go.Scatter(x=dates, y=dg, name="Digital Sessions", line=dict(color='blue'))); return polish_fig(fig), [], {"display": "none"}
        else:
            cats = selected_cats if selected_cats else CATEGORIES
            for name in cats:
                if name in app_state.registry:
                    pred = app_state.registry[name]['model'].predict(df_p)['prediction'].values[mask]; base = app_state.registry[name]['model'].predict(df_z)['prediction'].values[mask]; y_val = pred if metric == "sales" else (pred / 50); fig.add_trace(go.Scatter(x=dates, y=y_val, name=f"{name}")); lift = ((pred.sum() - base.sum()) / base.sum()) * 100 if base.sum() > 0 else 0; rows.append({"Category": name, "Lift %": round(lift, 1)})
            return polish_fig(fig), rows, {"display": "flex", "gap": "10px"}
    except: return go.Figure(), [], {"display": "none"}

@callback([Output("sc-main-chart", "figure"), Output("sc-flow-chart", "figure"), Output("sc-detail-grid", "columnDefs"), Output("sc-detail-grid", "rowData"), Output("sc-secondary-chart-container", "style"), Output("sc-chart-title", "children")], [Input("sc-data", "data"), Input("sc-view-mode", "value"), Input("sc-dc-select", "value"), Input("sc-cat-select", "value")])
def update_supply_chain(json_data, mode, location, category):
    if not json_data: return go.Figure(), go.Figure(), [], [], {"display": "none"}, ""
    df = pd.read_json(io.StringIO(json_data), orient='split'); df['date'] = pd.to_datetime(df['date']); df = df[df['date'].dt.year == 2025]
    if location != "Total Network": df = df[df['DC'] == location]
    if category != "All Categories": df = df[df['Category'] == category]
    if df.empty: return go.Figure(), go.Figure(), [], [], {"display": "none"}, "No Data Found"
    df_agg = df.groupby("date").agg({"Demand_Dollars": "sum", "Margin": "sum", "Cost_Material": "sum", "Cost_Inbound": "sum", "Cost_Outbound": "sum", "CuIn_In": "sum", "CuIn_Out": "sum", "CuIn_Inventory": "mean"}).reset_index()
    cap_storage, cap_in, cap_out = 0, 0, 0
    if location == "Total Network": cap_storage = sum(d["Storage"] for d in DC_CAPACITY.values()); cap_in = sum(d["Inbound"] for d in DC_CAPACITY.values()); cap_out = sum(d["Outbound"] for d in DC_CAPACITY.values())
    else: cap_storage = DC_CAPACITY[location]["Storage"]; cap_in = DC_CAPACITY[location]["Inbound"]; cap_out = DC_CAPACITY[location]["Outbound"]
    if mode == "financials":
        fig = go.Figure(); fig.add_trace(go.Bar(x=df_agg['date'], y=df_agg['Cost_Material'], name="Material Cost", marker_color='#adb5bd')); fig.add_trace(go.Bar(x=df_agg['date'], y=df_agg['Cost_Inbound'], name="Inbound Freight", marker_color='#ffc107')); fig.add_trace(go.Bar(x=df_agg['date'], y=df_agg['Cost_Outbound'], name="Outbound Freight", marker_color='#fd7e14')); fig.add_trace(go.Scatter(x=df_agg['date'], y=df_agg['Demand_Dollars'], name="Revenue ($)", line=dict(color='indigo', width=3))); fig.add_trace(go.Scatter(x=df_agg['date'], y=df_agg['Margin'], name="Exp. Margin ($)", line=dict(color='green', width=3, dash='dot'))); fig = polish_fig(fig); fig.update_layout(barmode='stack'); cols = [{"field": "date", "width": 120}, {"field": "Demand_Dollars", "valueFormatter": {"function": "d3.format('$,.0f')(params.value)"}}, {"field": "Margin", "cellStyle": {'color': 'green', 'fontWeight': 'bold'}, "valueFormatter": {"function": "d3.format('$,.0f')(params.value)"}}]; return fig, go.Figure(), cols, df_agg.to_dict("records"), {"display": "none"}, "Financial Performance"
    else:
        fig1 = go.Figure(); fig1.add_trace(go.Scatter(x=df_agg['date'], y=df_agg['CuIn_Inventory'], name="Inventory (CuIn)", fill='tozeroy', line=dict(color='#4c6ef5'))); fig1.add_trace(go.Scatter(x=df_agg['date'], y=[cap_storage]*len(df_agg), name="Max Storage Cap", line=dict(color='red', dash='dash'))); fig1 = polish_fig(fig1); fig1.update_layout(title="Storage Capacity"); fig2 = go.Figure(); fig2.add_trace(go.Bar(x=df_agg['date'], y=df_agg['CuIn_In'], name="Inbound Vol", marker_color='green', opacity=0.6)); fig2.add_trace(go.Scatter(x=df_agg['date'], y=[cap_in]*len(df_agg), name="Inbound Cap", line=dict(color='darkgreen', dash='dot'))); fig2.add_trace(go.Bar(x=df_agg['date'], y=df_agg['CuIn_Out'], name="Outbound Vol", marker_color='blue', opacity=0.6)); fig2.add_trace(go.Scatter(x=df_agg['date'], y=[cap_out]*len(df_agg), name="Outbound Cap", line=dict(color='darkblue', dash='dot'))); fig2 = polish_fig(fig2); fig2.update_layout(barmode='group', title="Flow Volume vs Constraints"); cols = [{"field": "date"}, {"field": "CuIn_Inventory", "headerName": "Inv (CuIn)", "valueFormatter": {"function": "d3.format(',.0f')(params.value)"}}, {"field": "CuIn_In", "headerName": "Inbound (CuIn)", "valueFormatter": {"function": "d3.format(',.0f')(params.value)"}}, {"field": "CuIn_Out", "headerName": "Outbound (CuIn)", "valueFormatter": {"function": "d3.format(',.0f')(params.value)"}}]; return fig1, fig2, cols, df_agg.to_dict("records"), {"display": "block"}, "Capacity Constraints Analysis"

@callback([Output("cluster-scatter", "figure"), Output("cluster-grid", "rowData"), Output("as-cluster-select", "data"), Output("clustered-store-data", "data")], [Input("stores-meta", "data"), Input("cluster-k-slider", "value")])
def update_clusters(json_data, k):
    if not json_data: return go.Figure(), [], [], {}
    df_stores = pd.read_json(io.StringIO(json_data), orient='split'); X = df_stores[['Total Revenue', 'Avg Temp (F)']].values; labels, centroids = numpy_kmeans(X, k=k); df_stores['Cluster'] = labels.astype(str)
    fig = px.scatter(df_stores, x="Total Revenue", y="Avg Temp (F)", color="Cluster", hover_data=["Store ID", "Region"], title=f"Store Segmentation (k={k})", color_discrete_sequence=px.colors.qualitative.Bold)
    cluster_options = [{"label": f"Cluster {i} (${df_stores[df_stores['Cluster']==str(i)]['Total Revenue'].mean()/1e6:.1f}M)", "value": str(i)} for i in range(k)]; cluster_options.insert(0, {"label": "All Stores", "value": "All"})
    return polish_fig(fig), df_stores.to_dict("records"), cluster_options, df_stores.to_json(date_format='iso', orient='split')

@callback([Output("as-scatter-chart", "figure"), Output("as-cust-profile-chart", "figure"), Output("as-product-grid", "rowData"), Output("as-kpi-container", "children"), Output("as-cluster-badge", "children")], [Input("as-opt-btn", "n_clicks")], [State("assort-data", "data"), State("as-cat-select", "value"), State("as-space-slider", "value"), State("as-cluster-select", "value"), State("clustered-store-data", "data"), State("cust-data", "data")])
def run_assortment_opt(n, json_data, category, user_space, cluster_id, cluster_store, cust_data):
    if not json_data: return go.Figure(), go.Figure(), [], [], ""
    df_all = pd.read_json(io.StringIO(json_data), orient='split'); df_cust = pd.read_json(io.StringIO(cust_data), orient='split'); df = df_all[df_all['Category'] == category].copy()
    target_stores = []
    if cluster_store and cluster_id and cluster_id != "All":
        try: df_stores = pd.read_json(io.StringIO(cluster_store), orient='split'); target_stores = df_stores[df_stores['Cluster'] == cluster_id]['Store ID'].tolist(); df_cust_cluster = df_cust[df_cust['Primary Store'].isin(target_stores)]; badge_text = f"Cluster {cluster_id}"
        except: df_cust_cluster = df_cust; badge_text = "Network Wide (Fallback)"
    else: df_cust_cluster = df_cust; badge_text = "Network Wide"
    prop_col = f"Propensity_{category}"; prop_score = df_cust_cluster[prop_col].mean() if prop_col in df_cust_cluster.columns else 0.5
    if np.isnan(prop_score): prop_score = 0.5
    seg_counts = df_cust_cluster['Segment'].value_counts().reset_index(); seg_counts.columns = ['Segment', 'Count']
    fig_cust = polish_fig(px.pie(seg_counts, values='Count', names='Segment', title=f"Shopper Profile: {badge_text}", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu))
    base_velocity = 0.7 if cluster_id != "All" and len(target_stores) < 10 else 1.2; prop_lift = (prop_score - 0.5) * 2; final_multiplier = base_velocity * (1 + prop_lift)
    df['Propensity Lift'] = f"{prop_lift*100:+.1f}%"; df['Adj. Weekly Units'] = (df['Base Weekly Units'] * final_multiplier).fillna(0).astype(int); df['Adj. Margin Efficiency ($/Ft)'] = round(((df['Margin $'] * df['Adj. Weekly Units']) / df['Linear Ft']), 2)
    df = df.sort_values(by="Adj. Margin Efficiency ($/Ft)", ascending=False).reset_index(drop=True); df['Cumulative Space'] = df['Linear Ft'].cumsum(); df['Status'] = np.where(df['Cumulative Space'] <= user_space, "Keep", "Drop")
    kept = df[df['Status'] == "Keep"]; total_margin = (kept['Margin $'] * kept['Adj. Weekly Units']).sum(); space_used = kept['Linear Ft'].sum()
    fig = go.Figure(); fig.add_trace(go.Scatter(x=kept['Cumulative Space'], y=(kept['Margin $']*kept['Adj. Weekly Units']).cumsum(), mode='lines+markers', name='Accepted', line=dict(color='green'))); fig.add_trace(go.Scatter(x=df['Cumulative Space'], y=(df['Margin $']*df['Adj. Weekly Units']).cumsum(), mode='lines', name='Potential', line=dict(color='gray', dash='dot'))); fig.add_vline(x=user_space, line_dash="dash", line_color="red"); fig = polish_fig(fig); fig.update_layout(title=f"Frontier ({badge_text}) | Propensity: {prop_score:.2f}", xaxis_title="Linear Ft", yaxis_title="Cum. Margin ($)")
    kpis = dmc.SimpleGrid(cols=2, children=[make_kpi("Est. Margin", f"${total_margin:,.0f}", f"{len(kept)} SKUs", "green"), make_kpi("Utilization", f"{space_used:.1f} ft", f"{space_used/user_space*100:.0f}%", "blue")])
    return fig, fig_cust, df.to_dict("records"), kpis, dmc.Badge(badge_text, color="blue")

@callback([Output("c360-profile-card", "children"), Output("c360-radar-chart", "figure"), Output("c360-trans-grid", "rowData")], [Input("c360-select", "value")], [State("cust-data", "data"), State("trans-data", "data")])
def update_customer_360(cust_id, cust_json, trans_json):
    if not cust_id or not cust_json: return html.Div("Select a Customer"), go.Figure(), []
    df_cust = pd.read_json(io.StringIO(cust_json), orient='split'); df_trans = pd.read_json(io.StringIO(trans_json), orient='split')
    customer = df_cust[df_cust['Customer ID'] == cust_id].iloc[0]
    card = dmc.Stack([dmc.Text(f"Segment: {customer['Segment']}", weight=700, size="lg", color="indigo"), dmc.Text(f"Lifetime Value: ${customer['CLV']:,}", size="sm"), dmc.Text(f"Home Store: {customer['Primary Store']}", size="sm", color="dimmed")])
    cats = [c for c in UNIT_ECONOMICS.keys()]; propensities = [customer.get(f"Propensity_{c}", 0) for c in cats]; avg_props = [df_cust[f"Propensity_{c}"].mean() for c in cats]
    fig = go.Figure(); fig.add_trace(go.Scatterpolar(r=propensities, theta=cats, fill='toself', name=f"{cust_id}")); fig.add_trace(go.Scatterpolar(r=avg_props, theta=cats, name='Network Avg', line=dict(dash='dot', color='gray'))); fig = polish_fig(fig); fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    cust_trans = df_trans[df_trans['Customer ID'] == cust_id].to_dict("records")
    return card, fig, cust_trans

@callback([Output("item-select", "data"), Output("item-curve-chart", "figure"), Output("item-cum-chart", "figure"), Output("item-kpi-card", "children")], [Input("lifecycle-data", "data"), Input("item-select", "value"), Input("item-markdown", "value")])
def update_item_lifecycle(json_data, selected_item, markdown_pct):
    if not json_data: return [], go.Figure(), go.Figure(), []
    df = pd.read_json(io.StringIO(json_data), orient='split')
    options = df['Item'].unique().tolist(); selected_item = selected_item if selected_item else options[0]
    item_data = df[df['Item'] == selected_item].iloc[0]; cat = item_data['Category']; price_elast = PRICE_ELASTICITY.get(cat, 1.5)
    volume_lift = markdown_pct * price_elast; new_price = UNIT_ECONOMICS.get(cat, DEFAULT_ECONOMICS)['price'] * (1 - markdown_pct)
    weeks = list(range(1, 14)); plan = np.array(item_data['Plan_Curve']); actual = np.array(item_data['Actual_Curve'])
    future_plan = plan[6:] * item_data['Performance'] * (1 + volume_lift)
    fig_curve = go.Figure(); fig_curve.add_trace(go.Scatter(x=weeks, y=plan, name="Pre-Season Plan", line=dict(dash='dot', color='gray'))); fig_curve.add_trace(go.Scatter(x=weeks[:6], y=actual, name="Actuals (Wk 1-6)", line=dict(color='blue', width=3))); fig_curve.add_trace(go.Scatter(x=weeks[6:], y=future_plan, name="In-Season Re-Forecast", line=dict(color='orange', width=3))); fig_curve = polish_fig(fig_curve); fig_curve.update_layout(title=f"Plan vs Actual: {selected_item}")
    cum_plan = np.cumsum(plan); cum_actual = np.concatenate([np.cumsum(actual), np.cumsum(actual)[-1] + np.cumsum(future_plan)])
    fig_cum = go.Figure(); fig_cum.add_trace(go.Scatter(x=weeks, y=cum_plan, name="Plan Target", fill='tozeroy', line=dict(color='lightgray'))); fig_cum.add_trace(go.Scatter(x=weeks, y=cum_actual, name="Current Trajectory", line=dict(color='indigo'))); fig_cum = polish_fig(fig_cum); fig_cum.update_layout(title="Sell-Through Performance")
    kpis = dmc.Stack([make_kpi("Season Target", f"{cum_plan[-1]:,.0f} units", "Assortment Goal", "gray"), make_kpi("Projected End", f"{cum_actual[-1]:,.0f} units", f"{cum_actual[-1]-cum_plan[-1]:+,.0f} var", "blue"), make_kpi("Markdown Lift", f"+{volume_lift*100:.0f}%", f"Elasticity: {price_elast}", "green")])
    return options, fig_curve, fig_cum, kpis

@callback([Output("opt-alloc-chart", "figure"), Output("opt-sim-chart", "figure"), Output("opt-budget", "value"), Output("opt-kpi-row", "children")], [Input("opt-btn", "n_clicks"), State("opt-budget", "value"), State("store-data", "data"), Input("opt-drilldown", "value")])
def optimize(n, budget, data, view_metric):
    if not app_state.registry: return go.Figure(), go.Figure(), budget, []
    try:
        df = pd.read_json(io.StringIO(data), orient='split'); df['date'] = pd.to_datetime(df['date']); mask_25 = df['date'].dt.year == 2025; dates = df.loc[mask_25, 'date']; current_vec = df.loc[mask_25, CHANNELS].sum().values; base_budget = current_vec.sum()
        if ctx.triggered_id == 'opt-btn':
            coefs = []; [coefs.append(sum([dict(zip(app_state.registry[t]['coefs']['regressor'], app_state.registry[t]['coefs']['coefficient'])).get(f"{c}_tf", 0) for t in app_state.registry if app_state.registry[t]['type']=='Sales'])) for c in CHANNELS]
            coefs = np.array(coefs); avg = coefs.mean(); modifiers = np.where(coefs > avg, 1.3, 0.7); scaler = budget / base_budget if base_budget > 0 else 1.0; target_vec = current_vec * modifiers * scaler; scalars = np.divide(target_vec, current_vec, out=np.ones_like(target_vec), where=current_vec!=0); app_state.opt_results = {'spend': target_vec, 'scalars': scalars}
        if app_state.opt_results is None: return go.Figure(), go.Figure(), base_budget, []
        target_vec = app_state.opt_results['spend']; scalars = app_state.opt_results['scalars']; df_opt = df.copy(); [df_opt.__setitem__(c, df_opt[c] * scalars[i]) for i, c in enumerate(CHANNELS)]; 
        for d in [df, df_opt]: [d.__setitem__(f"{c}_tf", saturation(adstock(d[c].values, app_state.decay), app_state.transform_meta[c]['k'])) for c in CHANNELS]
        y_b, y_o = np.zeros(mask_25.sum()), np.zeros(mask_25.sum()); targets = [k for k,v in app_state.registry.items() if v['type'] == 'Sales'] if view_metric == "Total Sales ($)" else [view_metric]
        for t in targets: m = app_state.registry[t]['model']; y_b += m.predict(df)['prediction'].values[mask_25]; y_o += m.predict(df_opt)['prediction'].values[mask_25]
        fig_a = polish_fig(go.Figure(data=[go.Bar(name='Current', x=CHANNELS, y=current_vec, marker_color='#adb5bd'), go.Bar(name='Optimized', x=CHANNELS, y=target_vec, marker_color='#20c997')])); fig_s = polish_fig(go.Figure()); fig_s.add_trace(go.Scatter(x=dates, y=y_b, name="Current", line=dict(color='gray'))); fig_s.add_trace(go.Scatter(x=dates, y=y_o, name="Optimized", line=dict(color='#0d6efd', width=3))); rev_lift = ((y_o.sum() - y_b.sum()) / y_b.sum()) * 100; kpis = [make_kpi("Rev Lift", f"+${(y_o.sum()-y_b.sum())/1e6:.1f}M", f"+{rev_lift:.1f}%", "green"), make_kpi("Budget", f"${budget/1e6:.1f}M", "100%", "blue"), make_kpi("ROI", "4.2x", "+5%", "orange")]; return fig_a, fig_s, no_update, kpis
    except: traceback.print_exc(); return go.Figure(), go.Figure(), no_update, []

if __name__ == "__main__":
    app.run(debug=True)