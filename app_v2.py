import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ==========================================
# 1. Page Config
# ==========================================
st.set_page_config(
    page_title="California House Pricing",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. Custom CSS (Premium Modern Design)
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700&display=swap');

    /* Variables and Globals */
    :root {
        --primary: #6366f1;
        --secondary: #10b981;
        --accent: #f59e0b;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --border: #334155;
    }

    .stApp {
        background-color: var(--bg-dark);
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: var(--text-main) !important;
    }

    /* Sidebar cleanup */
    [data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid var(--border);
    }

    /* Tabs Styling - Clean and Minimal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
        padding: 0.5rem 0;
        margin-bottom: 2rem;
        border: none !important;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        color: var(--text-muted);
        font-weight: 600;
        font-size: 1rem;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--primary) !important;
        background-color: transparent !important;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--primary) !important;
    }

    /* Analysis Summary (Restored Style) */
    .analysis-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 1.2rem;
        font-family: 'Outfit', sans-serif;
    }
    .status-metric {
        padding: 0.5rem 0;
    }
    .status-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-main);
        line-height: 1.1;
        margin: 4px 0;
        font-family: 'Outfit', sans-serif;
    }
    .status-sub {
        font-size: 0.75rem;
        color: var(--secondary);
        display: flex;
        align-items: center;
        gap: 4px;
        margin-top: 6px;
    }
    .status-sub.gold { color: var(--accent); }

    /* Cards and Containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }

    .prediction-hero {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid var(--primary);
        border-radius: 24px;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 25px 50px -12px rgba(99, 102, 241, 0.25);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    .price-value {
        font-size: 4rem;
        font-weight: 800;
        color: var(--primary);
        font-family: 'Outfit', sans-serif;
        margin: 0.5rem 0;
        letter-spacing: -2px;
    }

    .price-label {
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .metric-box-new {
        background: #1e293b;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-box-new:hover {
        border-color: var(--primary);
        transform: translateY(-4px);
    }

    .metric-val {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--secondary);
        font-family: 'Outfit', sans-serif;
    }

    .metric-lbl {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* Banner types */
    .banner {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .banner-warn { background: rgba(245, 158, 11, 0.1); border-left: 4px solid var(--accent); color: #fbbf24; }
    .banner-info { background: rgba(99, 102, 241, 0.1); border-left: 4px solid var(--primary); color: #a5b4fc; }

    /* Inputs */
    .stNumberInput input, .stSelectbox select {
        background-color: #0f172a !important;
        border: 1px solid var(--primary) !important;
        color: white !important;
        border-radius: 8px !important;
    }

    .stNumberInput input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
        outline: none !important;
    }
    
    /* Plotly Chart refinement */
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
    }
    
    .arabic-hint {
        color: var(--text-muted);
        font-size: 0.85rem;
        font-style: italic;
    }

    /* ── Number Input Buttons ── */
    .stNumberInput button {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-main) !important;
        border-radius: 6px !important;
        transition: background-color 0.2s ease, border-color 0.2s ease !important;
        cursor: pointer !important;
    }

    /* زرار + (آخر زرار) — أخضر عند hover */
    .stNumberInput button:last-of-type:hover {
        background-color: #10b981 !important;
        border-color: #10b981 !important;
        color: white !important;
    }

    /* زرار - (أول زرار) — أحمر عند hover */
    .stNumberInput button:first-of-type:hover {
        background-color: #ef4444 !important;
        border-color: #ef4444 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. Helpers, Constants, Data & Model Loading
# ==========================================
OCEAN_CATS = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

FEATURE_ORDER = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
    'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
    'ocean_proximity_NEAR OCEAN',
    'rooms_per_household', 'bedrooms_per_room',
    'population_per_household', 'income_per_room',
]

MODEL_METRICS = {
    "R² Score": ("0.837", "High accuracy"),
    "RMSE":     ("$42,600", "Avg. Deviation"),
    "MAE":      ("$29,800", "Median Error"),
}

FEATURE_IMPORTANCE = {
    "median_income":              0.421,
    "rooms_per_household":        0.118,
    "ocean_proximity_INLAND":     0.094,
    "latitude":                   0.071,
    "longitude":                  0.065,
    "housing_median_age":         0.058,
    "bedrooms_per_room":          0.047,
    "population_per_household":   0.041,
    "income_per_room":            0.034,
    "households":                 0.021,
    "total_rooms":                0.018,
    "ocean_proximity_NEAR BAY":   0.012,
}

# Fixed dataset medians — computed once from the original California Housing dataset
# using pandas. These values must NEVER be derived dynamically from user inputs.
DATASET_MEDIANS = {
    "median_income":              3.5348,   # in $10k units
    "housing_median_age":        29.0,      # years
    "rooms_per_household":        5.2291,
    "population_per_household":   2.8181,
    "bedrooms_per_room":          0.2031,   # raw ratio (×10 in radar for scale → 2.0316)
}

@st.cache_resource
def load_model():
    json_path = "xgboost_housing_model.json"
    pkl_path  = "xgboost_housing_model.pkl"
    if os.path.exists(json_path):
        m = xgb.XGBRegressor()
        m.load_model(json_path)
        return m, True
    if os.path.exists(pkl_path):
        return joblib.load(pkl_path), True
    return None, False

@st.cache_data
def load_data():
    data_path = 'housing.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

def build_sample(longitude, latitude, age, total_rooms, total_bedrooms,
                 population, households, median_income, ocean_proximity):
    op_dummies = {f"ocean_proximity_{c}": 0 for c in OCEAN_CATS}
    op_dummies[f"ocean_proximity_{ocean_proximity}"] = 1

    row = {
        "longitude":             longitude,
        "latitude":              latitude,
        "housing_median_age":    age,
        "total_rooms":           total_rooms,
        "total_bedrooms":        total_bedrooms,
        "population":            population,
        "households":            households,
        "median_income":         median_income,
        **op_dummies,
        "rooms_per_household":   total_rooms / households if households else 0,
        "bedrooms_per_room":     total_bedrooms / total_rooms if total_rooms else 0,
        "population_per_household": population / households if households else 0,
        "income_per_room":       median_income / total_rooms if total_rooms else 0,
    }
    df = pd.DataFrame([row])
    return df.reindex(columns=FEATURE_ORDER, fill_value=0)

def predict(model, df_sample):
    pred_log = model.predict(df_sample.values)
    return np.expm1(pred_log)[0]

model, model_loaded = load_model()
df = load_data()

# Initialize session state
if 'median_income' not in st.session_state: st.session_state.median_income = 8.3252
if 'age' not in st.session_state: st.session_state.age = 41.0
if 'rooms_ph' not in st.session_state: st.session_state.rooms_ph = 6.98
if 'pop_ph' not in st.session_state: st.session_state.pop_ph = 2.55
if 'bed_ratio' not in st.session_state: st.session_state.bed_ratio = 0.146

# ==========================================
# 4. Sidebar Content
# ==========================================
with st.sidebar:
    st.image("icon.png", width=200)
    st.title("Settings")
    st.markdown("---")
    st.markdown("### 🛠 Options")
    debug_mode = st.toggle("Debug Info", value=False)
    st.markdown("---")
    st.info("💡 **Tip:** Adjust location on the map tab to auto-update predictions.")

# ==========================================
# 5. Main Navigation (Using Tabs)
# ==========================================
tab_pred, tab_insights, tab_data, tab_about = st.tabs([
    "Price Predictor", 
    "Model Insights", 
    "🗺 Data Explorer", 
    "About ?"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: PRICE PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab_pred:
    st.markdown("# 🏠 Real Estate Value Predictor")
    st.markdown("Leverage AI to estimate California housing prices with precision. <span class='arabic-hint'>(توقع أسعار العقارات في كاليفورنيا بدقة الذكاء الاصطناعي)</span>", unsafe_allow_html=True)
    
    # Input Area
    with st.container():
        col_loc, col_specs, col_income = st.columns(3)

        with col_loc:
            st.markdown(""" <div style='color:#f59e0b; font-size:1.7rem; font-weight:600; margin-bottom:0.5rem;'>Location Details</div>
                        """, unsafe_allow_html=True )
            longitude = st.number_input("Longitude (خط الطول)", min_value=-125.0, max_value=-114.0, value=-122.23, step=0.1)
            latitude  = st.number_input("Latitude (خط العرض)", min_value=32.0, max_value=42.0, value=37.88, step=0.1)

        # Auto-detection — must run before col_specs & col_income so def_ variables are defined
        if df is not None:
            dists = (df['latitude'] - latitude)**2 + (df['longitude'] - longitude)**2
            closest_idx = dists.idxmin()
            ocean_proximity = df.loc[closest_idx, 'ocean_proximity']
            def_age    = float(df.loc[closest_idx, 'housing_median_age'])
            def_pop    = float(df.loc[closest_idx, 'population'])
            def_hh     = float(df.loc[closest_idx, 'households'])
            def_rooms  = float(df.loc[closest_idx, 'total_rooms'])
            def_beds   = float(df.loc[closest_idx, 'total_bedrooms'])
            if pd.isna(def_beds): def_beds = max(1.0, float(def_rooms * 0.2))
            def_income = float(df.loc[closest_idx, 'median_income'])
            def_income = min(15.0, max(0.0, def_income))
        else:
            ocean_proximity = "NEAR BAY"
            def_age, def_pop, def_hh, def_rooms, def_beds, def_income = 41.0, 322.0, 126.0, 880.0, 129.0, 8.3252

        with col_specs:
            st.markdown(""" <div style='color:#f59e0b; font-size:1.7rem; font-weight:600; margin-bottom:0.5rem;'>Property Specs</div>
                        """, unsafe_allow_html=True )
            inp_rooms = st.number_input("Rooms", min_value=1, max_value=10000, value=int(def_rooms), step=10)
            inp_beds  = st.number_input("Beds",  min_value=1, max_value=int(inp_rooms), value=min(int(def_beds), int(inp_rooms)), step=5)
            st.markdown(f"<span style='color:#22c55e;'>Suggested: {int(def_rooms):,} rooms · {int(def_beds):,} beds</span>", unsafe_allow_html=True)

        with col_income:
            st.markdown(""" <div style='color:#f59e0b; font-size:1.7rem; font-weight:600; margin-bottom:0.5rem;'>Median Income</div>
                        """, unsafe_allow_html=True )
            inp_income = st.number_input("Median Income ($10k)", min_value=0.0, max_value=15.0, value=float(def_income), step=0.1)
            st.markdown(f"<span style='color:#22c55e;'>Suggested: ${def_income:.2f}k</span>", unsafe_allow_html=True)

        # Summary Row
        s1, s2, s3, space = st.columns([1,1,1,1.5])

        with s1:
            st.markdown(f"""<div class='status-metric'>
                <div class='status-label'>Proximity Status - القرب</div>
                <div class='status-value'>{ocean_proximity}</div>
                <div class='status-sub'>↑ Nearest Land Point</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class='status-metric'>
                <div class='status-label'>Median Age - عمر المباني</div>
                <div class='status-value'>{def_age:.0f} Yrs</div>
                <div class='status-sub'>District Average</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""<div class='status-metric'>
                <div class='status-label'>Population - السكان</div>
                <div class='status-value'>{def_pop:,.0f}</div>
                <div class='status-sub gold'>Pop / {def_hh:,.0f} HH</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        st.markdown(
    """
    <div style="
        height:1px;
        width:100%;
        background: #64748b;
        box-shadow: 0 0 6px #64748b;
        opacity: 0.6;
        margin: 18px 0 25px 0;
    "></div>
    """, unsafe_allow_html=True )

    # Sync Session State — locked values from location, open values from user inputs
    st.session_state.median_income = inp_income
    st.session_state.age = def_age
    st.session_state.rooms_ph = inp_rooms / def_hh if def_hh > 0 else 0
    st.session_state.pop_ph = def_pop / def_hh if def_hh > 0 else 0
    st.session_state.bed_ratio = inp_beds / inp_rooms if inp_rooms > 0 else 0

    # Main Output Section
    col_out, col_map = st.columns([1, 1.2], gap="large")
    
    with col_out:
        if not model_loaded:
            st.markdown("<div class='banner banner-warn'>⚠️ Model file missing! Prediction unavailable.</div>", unsafe_allow_html=True)
        
        # Validation
        is_valid = True
        if df is not None:
            min_dist = np.sqrt((df['latitude'] - latitude)**2 + (df['longitude'] - longitude)**2).min()
            if min_dist > 0.3:
                st.markdown("<div class='banner banner-warn'>🌊 Location appears to be offshore. Please adjust.</div>", unsafe_allow_html=True)
                is_valid = False
        
        if inp_beds > inp_rooms:
            st.markdown("<div class='banner banner-warn'>⚠️ Logical error: Bedrooms > Total Rooms.</div>", unsafe_allow_html=True)
            is_valid = False

        if is_valid and model_loaded:
            sample = build_sample(longitude, latitude, def_age, inp_rooms, inp_beds, def_pop, def_hh, inp_income, ocean_proximity)
            price = predict(model, sample)
            low, high = price * 0.88, price * 1.12
            
            st.markdown(f"""
            <div class='prediction-hero'>
                <div class='price-label'>Estimated Median House Value</div>
                <div class='price-value'>${price:,.0f}</div>
                <div style='color:var(--secondary); font-weight:600;'>Range: ${low:,.0f} – ${high:,.0f}</div>
                <div style='color:var(--text-muted); font-size:0.75rem; margin-top:10px;'>Confidence Level: ±12% based on historical data</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='prediction-hero'>
                <div class='price-label'>Estimated Value</div>
                <div class='price-value'>$—</div>
                <div style='color:var(--text-muted);'>Waiting for valid inputs...</div>
            </div>
            """, unsafe_allow_html=True)

        # Detailed Metrics
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.markdown(f"<div class='metric-box-new'><div class='metric-val'>{st.session_state.rooms_ph:.1f}</div><div class='metric-lbl'>Rooms / HH</div></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-box-new'><div class='metric-val'>{st.session_state.pop_ph:.1f}</div><div class='metric-lbl'>People / HH</div></div>", unsafe_allow_html=True)
        with m_c2:
            st.markdown(f"<div class='metric-box-new'><div class='metric-val'>{st.session_state.bed_ratio:.3f}</div><div class='metric-lbl'>Bedrooms / Room</div></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-box-new'><div class='metric-val'>{inp_income / inp_rooms:.4f}</div><div class='metric-lbl'>Income / Room</div></div>", unsafe_allow_html=True)

    with col_map:
        st.markdown("### 📍 Location View")
        fig_map = go.Figure(go.Scattermapbox(
            lat=[latitude], lon=[longitude],
            mode="markers",
            marker=go.scattermapbox.Marker(size=20, color="#6366f1", opacity=0.8),
            text=[f"District Hub"],
            hoverinfo="text",
        ))
        fig_map.update_layout(
            mapbox=dict(style="carto-darkmatter", center=dict(lat=latitude, lon=longitude), zoom=8),
            margin=dict(l=0, r=0, t=0, b=0),
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown(f"<div class='banner banner-info'>📌 Selected District: {ocean_proximity} area at ({latitude:.2f}, {longitude:.2f})</div>", unsafe_allow_html=True)

    # ── Debug Section ──
    if debug_mode:
        st.markdown("---")
        st.markdown("### 🛠 Developer Debug Console")
        with st.expander("Raw Feature Vector (Model Input)", expanded=True):
            if is_valid and model_loaded:
                st.dataframe(sample, use_container_width=True)
                st.json(sample.to_dict(orient='records')[0])
            else:
                st.warning("No valid sample data available for debugging.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_insights:
    st.markdown("# Model Analytics")
    st.markdown("Understand how the AI makes decisions and its performance metrics.")
    
    m_cols = st.columns(3)
    for i, (k, (v, sub)) in enumerate(MODEL_METRICS.items()):
        with m_cols[i]:
            st.markdown(f"""
            <div class='metric-box-new'>
                <div class='metric-val' style='color:var(--primary)'>{v}</div>
                <div class='metric-lbl'>{k}</div>
                <div style='font-size:0.7rem; color:var(--text-muted);'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="
        height:1px;
        width:100%;
        background: #64748b;
        box-shadow: 0 0 6px #64748b;
        opacity: 0.6;
        margin: 18px 0 25px 0;
    "></div> """, unsafe_allow_html=True )
    c_fi, c_radar = st.columns([1, 1], gap="large")
    
    with c_fi:
        st.markdown("### 🔝 Feature Importance")
        fi_df = pd.DataFrame(list(FEATURE_IMPORTANCE.items()), columns=["Feature", "Importance"]).sort_values("Importance")
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation='h',
                        color="Importance", color_continuous_scale="Viridis")
        fig_fi.update_layout(
            height=400, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with c_radar:
        st.markdown("### 🕸 Input vs Dataset")
        # Radar Data — user values vs fixed dataset medians (DATASET_MEDIANS constant)
        comparisons = {
            "Income":    (st.session_state.median_income,   DATASET_MEDIANS["median_income"],                       0, 15),
            "Age":       (st.session_state.age,             DATASET_MEDIANS["housing_median_age"],                  1, 52),
            "Rooms/HH":  (st.session_state.rooms_ph,        DATASET_MEDIANS["rooms_per_household"],                 1, 20),
            "Pop/HH":    (st.session_state.pop_ph,          DATASET_MEDIANS["population_per_household"],            1, 20),
            "Bed Ratio": (st.session_state.bed_ratio * 10,  DATASET_MEDIANS["bedrooms_per_room"] * 10,              0, 10),
        }
        cats = list(comparisons.keys())
        u_vals = [min(1, max(0, (uv - lo) / (hi - lo))) for _, (uv, mv, lo, hi) in comparisons.items()]
        m_vals = [min(1, max(0, (mv - lo) / (hi - lo))) for _, (uv, mv, lo, hi) in comparisons.items()]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=u_vals + [u_vals[0]], theta=cats + [cats[0]], fill="toself", name="Current", line=dict(color="#6366f1")))
        fig_radar.add_trace(go.Scatterpolar(r=m_vals + [m_vals[0]], theta=cats + [cats[0]], fill="toself", name="Median", line=dict(color="#10b981", dash="dot")))
        fig_radar.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False, range=[0, 1])),
            paper_bgcolor="rgba(0,0,0,0)", height=400,
            font=dict(family="Inter", color="#94a3b8"), margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("# 🗺 Geographic Data Explorer")
    if df is not None:
        st.dataframe(df.head(100), use_container_width=True)

        
        st.markdown("### 🌎 Dataset Distribution")
        st.map(df[['latitude', 'longitude']].dropna(), zoom=5, color='#6366f1')
    else:
        st.error("Data source 'housing.csv' not found.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("# Project Documentation")
    
    col_a1, col_a2 = st.columns([1.5, 1])
    with col_a1:
        st.markdown("""
        ### Overview
        This tool implements an **XGBoost Regressor** to predict median house values. It processes 17 features including geographic coordinates, demographic data, and engineered ratios to provide accurate estimates.

        ### Methodology
        - **Model:** XGBoost with log-transformed target variable.
        - **Training:** Optimized via Random Search with 5-fold cross-validation.
        - **Features:** 8 raw, 5 proximity indicators, and 4 calculated ratios.
        """)
        
        st.markdown("### 👥 The Team")
        st.info("- Mohamed Hisham Sukar\n- Adham Abd-Rahman Mahmoud\n- Mohamed Sobhy Mohamed Foaud\n- AbdolRahman Mohamed Asem\n- Mina Bassem Samir Daniel")
        
    with col_a2:
        st.markdown("#### Tech Stack")
        st.code("Python 3.x\nStreamlit\nXGBoost\nPlotly\nPandas", language="text")
        st.markdown("#### Performance")
        st.progress(0.837, text="R² Score: 0.837")
