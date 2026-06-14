import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================
# 1. MEMBERSHIP FUNCTION PRIMITIVES
# ==========================================
def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / (b - a), 0.0)
    right = np.where(c > b, (c - x) / (c - b), 0.0)
    return np.clip(np.minimum(left, right), 0.0, 1.0)

def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / (b - a), 1.0)
    right = np.where(d > c, (d - x) / (d - c), 1.0)
    result = np.minimum(np.minimum(left, 1.0), right)
    return np.clip(result, 0.0, 1.0)

# ==========================================
# 2. LINGUISTIC VARIABLES
# ==========================================
INPUT_MFS = {
    'Palm_Price': {
        'Rendah': lambda x: trapmf(x, 3000, 3000, 3700, 4200),
        'Sedang': lambda x: trimf(x, 3800, 4500, 5500),
        'Tinggi': lambda x: trapmf(x, 5000, 6000, 8500, 8500),
    },
    'Palm_Change': {
        'Turun': lambda x: trapmf(x, -11, -11, -2.5, 0.0),
        'Stabil': lambda x: trimf(x, -2.0, 0, 2.0),
        'Naik': lambda x: trapmf(x, 0.0, 2.5, 12, 12),
    },
    'Crude_Price': {
        'Rendah': lambda x: trapmf(x, 60, 60, 73, 80),
        'Sedang': lambda x: trimf(x, 73, 85, 97),
        'Tinggi': lambda x: trapmf(x, 90, 100, 125, 125),
    },
    'Crude_Change': {
        'Turun': lambda x: trapmf(x, -13, -13, -2.5, 0.0),
        'Stabil': lambda x: trimf(x, -2, 0, 2.5),
        'Naik': lambda x: trapmf(x, 0, 2.5, 10, 10),
    },
    'Sugar_Price': {
        'Rendah': lambda x: trapmf(x, 16, 16, 18.5, 20.5),
        'Sedang': lambda x: trimf(x, 19, 22, 25),
        'Tinggi': lambda x: trapmf(x, 23.5, 26, 30, 30),
    }
}

OUTPUT_MFS = {
    'Sangat_Rendah': lambda x: trapmf(x, 13000, 13000, 15000, 16500),
    'Rendah':        lambda x: trimf(x, 15500, 16800, 18200),
    'Sedang':        lambda x: trimf(x, 17500, 19000, 21000),
    'Tinggi':        lambda x: trimf(x, 19500, 22000, 25000),
    'Sangat_Tinggi': lambda x: trapmf(x, 23000, 27900, 37000, 37000),
}

OUTPUT_SINGLETONS = {
    'Sangat_Rendah': 15000, 'Rendah': 16800, 'Sedang': 19000,
    'Tinggi': 22000, 'Sangat_Tinggi': 28000
}

OUTPUT_RANGE = np.linspace(13000, 37000, 800)

RULES = [
    ('Rendah', 'Stabil', 'Rendah', 'Stabil', 'Rendah', 'Sangat_Rendah'),
    ('Rendah', 'Stabil', 'Sedang', 'Stabil', 'Rendah', 'Rendah'),
    ('Sedang', 'Stabil', 'Sedang', 'Stabil', 'Sedang', 'Sedang'),
    ('Tinggi', 'Stabil', 'Sedang', 'Stabil', 'Sedang', 'Tinggi'),
    ('Tinggi', 'Naik',  'Tinggi', 'Naik',   'Tinggi', 'Sangat_Tinggi'),
    ('Sedang', 'Naik',  'Sedang', 'Naik',   'Sedang', 'Tinggi'),
    ('Sedang', 'Turun', 'Rendah', 'Turun',  'Rendah', 'Rendah'),
    ('Rendah', 'Turun', 'Rendah', 'Turun',  'Rendah', 'Sangat_Rendah'),
    ('Sedang', 'Naik',  'Sedang', 'Stabil', 'Sedang', 'Sedang'),
    ('Tinggi', 'Naik',  'Tinggi', 'Naik',   'Tinggi', 'Sangat_Tinggi'),
    ('Tinggi', 'Stabil','Sedang', 'Stabil', 'Tinggi', 'Sangat_Tinggi'),
    ('Rendah', 'Stabil','Rendah', 'Stabil', 'Rendah', 'Sangat_Rendah'),
    ('Sedang', 'Stabil','Tinggi', 'Stabil', 'Sedang', 'Tinggi'),
    ('Sedang', 'Stabil','Rendah', 'Stabil', 'Tinggi', 'Sedang'),
    ('Rendah', 'Turun', 'Rendah', 'Turun',  'Rendah', 'Sangat_Rendah'),
    ('Sedang', 'Stabil','Tinggi', 'Naik',   'Rendah', 'Tinggi'),
    ('Rendah', 'Stabil','Tinggi', 'Stabil', 'Sedang', 'Sedang'),
    ('Tinggi', 'Stabil','Rendah', 'Stabil', 'Sedang', 'Sedang'),
    ('Sedang', 'Naik',  'Sedang', 'Naik',   'Tinggi', 'Sangat_Tinggi'),
    ('Rendah', 'Turun', 'Sedang', 'Stabil', 'Rendah', 'Rendah'),
]

# ==========================================
# 3. ENGINE LOGIC
# ==========================================
def fuzzify(value, mf_dict):
    return {label: float(mf_fn(value)) for label, mf_fn in mf_dict.items()}

def defuzzify_centroid(aggregated_output, x_range):
    numerator = np.sum(x_range * aggregated_output)
    denominator = np.sum(aggregated_output)
    return numerator / denominator if denominator > 1e-10 else np.mean(x_range)

def predict_mamdani(palm_price, palm_change, crude_price, crude_change, sugar_price):
    mu = {
        'Palm_Price':  fuzzify(palm_price,  INPUT_MFS['Palm_Price']),
        'Palm_Change': fuzzify(palm_change, INPUT_MFS['Palm_Change']),
        'Crude_Price': fuzzify(crude_price, INPUT_MFS['Crude_Price']),
        'Crude_Change':fuzzify(crude_change,INPUT_MFS['Crude_Change']),
        'Sugar_Price': fuzzify(sugar_price, INPUT_MFS['Sugar_Price']),
    }
    aggregated = np.zeros(len(OUTPUT_RANGE))
    rule_strengths = []
    for rule in RULES:
        p_lbl, pc_lbl, c_lbl, cc_lbl, s_lbl, out_lbl = rule
        fs = min(
            mu['Palm_Price'].get(p_lbl, 0), mu['Palm_Change'].get(pc_lbl, 0),
            mu['Crude_Price'].get(c_lbl, 0), mu['Crude_Change'].get(cc_lbl, 0),
            mu['Sugar_Price'].get(s_lbl, 0)
        )
        rule_strengths.append(fs)
        if fs > 0:
            clipped = np.minimum(fs, OUTPUT_MFS[out_lbl](OUTPUT_RANGE))
            aggregated = np.maximum(aggregated, clipped)
    result = defuzzify_centroid(aggregated, OUTPUT_RANGE)
    return result, aggregated, mu, rule_strengths

def predict_sugeno(palm_price, palm_change, crude_price, crude_change, sugar_price):
    mu = {
        'Palm_Price':  fuzzify(palm_price,  INPUT_MFS['Palm_Price']),
        'Palm_Change': fuzzify(palm_change, INPUT_MFS['Palm_Change']),
        'Crude_Price': fuzzify(crude_price, INPUT_MFS['Crude_Price']),
        'Crude_Change':fuzzify(crude_change,INPUT_MFS['Crude_Change']),
        'Sugar_Price': fuzzify(sugar_price, INPUT_MFS['Sugar_Price']),
    }
    numerator, denominator = 0.0, 0.0
    rule_strengths = []
    for rule in RULES:
        p_lbl, pc_lbl, c_lbl, cc_lbl, s_lbl, out_lbl = rule
        fs = min(
            mu['Palm_Price'].get(p_lbl, 0), mu['Palm_Change'].get(pc_lbl, 0),
            mu['Crude_Price'].get(c_lbl, 0), mu['Crude_Change'].get(cc_lbl, 0),
            mu['Sugar_Price'].get(s_lbl, 0)
        )
        rule_strengths.append(fs)
        numerator += fs * OUTPUT_SINGLETONS[out_lbl]
        denominator += fs
    result = numerator / denominator if denominator > 1e-10 else np.mean(list(OUTPUT_SINGLETONS.values()))
    return result, mu, rule_strengths

# ==========================================
# 4. PLOT HELPERS
# ==========================================
PALETTE = {
    'Rendah':        '#34C759',
    'Turun':         '#34C759',
    'Sedang':        '#007AFF',
    'Stabil':        '#007AFF',
    'Tinggi':        '#FF9500',
    'Naik':          '#FF9500',
    'Sangat_Rendah': '#5856D6',
    'Sangat_Tinggi': '#FF3B30',
}

def mf_color(label):
    for k, v in PALETTE.items():
        if k in label:
            return v
    return '#8E8E93'

def plot_membership_functions(var_name, mf_dict, current_value, x_range):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F2F2F7')
    for label, mf_fn in mf_dict.items():
        y = mf_fn(x_range)
        color = mf_color(label)
        ax.plot(x_range, y, label=label, color=color, linewidth=2)
        mu_val = float(mf_fn(current_value))
        if mu_val > 0:
            ax.fill_between(x_range, 0, np.minimum(mu_val, y), color=color, alpha=0.18)
    ax.axvline(x=current_value, color='#1C1C1E', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel(var_name, fontsize=9, color='#1C1C1E')
    ax.set_ylabel('μ', fontsize=9, color='#1C1C1E')
    ax.tick_params(colors='#3A3A3C', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#C7C7CC')
    ax.spines['bottom'].set_color('#C7C7CC')
    legend = ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85)
    fig.tight_layout(pad=1.0)
    return fig

def plot_output_aggregation(aggregated, result):
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F2F2F7')
    ax.fill_between(OUTPUT_RANGE, 0, aggregated, color='#007AFF', alpha=0.35, label='Aggregated Output')
    ax.plot(OUTPUT_RANGE, aggregated, color='#007AFF', linewidth=1.5)
    ax.axvline(x=result, color='#FF3B30', linewidth=2.0, linestyle='--', label=f'CoG: IDR {result:,.0f}')
    ax.set_xlabel('Cooking Oil Price (IDR/liter)', fontsize=9, color='#1C1C1E')
    ax.set_ylabel('μ', fontsize=9, color='#1C1C1E')
    ax.set_title('Mamdani — Aggregated Output & Defuzzification', fontsize=10, color='#1C1C1E', pad=8)
    ax.tick_params(colors='#3A3A3C', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#C7C7CC')
    ax.spines['bottom'].set_color('#C7C7CC')
    ax.legend(fontsize=8, framealpha=0.85)
    fig.tight_layout(pad=1.0)
    return fig

def plot_rule_activations(rule_strengths):
    active = [(i+1, s) for i, s in enumerate(rule_strengths) if s > 0]
    if not active:
        return None
    idxs, strengths = zip(*active)
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(active) * 0.4)))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F2F2F7')
    colors = ['#007AFF' if s >= 0.5 else '#5AC8FA' if s >= 0.2 else '#C7C7CC' for s in strengths]
    bars = ax.barh([f'Rule {i}' for i in idxs], strengths, color=colors, height=0.6)
    for bar, val in zip(bars, strengths):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=7.5, color='#1C1C1E')
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Firing Strength', fontsize=9, color='#1C1C1E')
    ax.set_title('Active Rule Firing Strengths', fontsize=10, color='#1C1C1E', pad=8)
    ax.tick_params(colors='#3A3A3C', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#C7C7CC')
    ax.spines['bottom'].set_color('#C7C7CC')
    fig.tight_layout(pad=1.0)
    return fig

# ==========================================
# 5. STREAMLIT UI
# ==========================================
st.set_page_config(
    page_title="FuzzyOil — Cooking Oil Price Prediction",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    /* Main font & background */
    html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', sans-serif; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        color: white;
    }
    .hero-banner h1 { font-size: 2rem; font-weight: 700; margin: 0 0 6px 0; letter-spacing: -0.5px; color: white; }
    .hero-banner p  { font-size: 0.95rem; margin: 0; opacity: 0.88; color: white; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.04);
        text-align: center;
        border: 1px solid #F2F2F7;
    }
    .metric-card .label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #8E8E93; margin-bottom: 8px; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1C1C1E; letter-spacing: -1px; line-height: 1; }
    .metric-card .sub   { font-size: 0.78rem; color: #8E8E93; margin-top: 6px; }

    /* Section headers */
    .section-header {
        font-size: 1.05rem; font-weight: 700; color: #1C1C1E;
        border-left: 3px solid #007AFF; padding-left: 10px;
        margin: 20px 0 14px 0;
    }

    /* Tag pill */
    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 2px;
    }
    .pill-blue   { background: #E1F0FF; color: #007AFF; }
    .pill-green  { background: #E5F8ED; color: #34C759; }
    .pill-orange { background: #FFF3E0; color: #FF9500; }
    .pill-red    { background: #FFE5E5; color: #FF3B30; }
    .pill-purple { background: #F0EEFF; color: #5856D6; }

    /* Fuzzification table */
    .fuzz-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    .fuzz-table th { background: #F2F2F7; padding: 7px 10px; text-align: left; font-weight: 600; color: #3A3A3C; border-bottom: 1px solid #E5E5EA; }
    .fuzz-table td { padding: 6px 10px; border-bottom: 1px solid #F2F2F7; color: #1C1C1E; }
    .fuzz-bar { height: 6px; border-radius: 3px; background: #E5E5EA; }
    .fuzz-fill { height: 6px; border-radius: 3px; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #F2F2F7; }
    section[data-testid="stSidebar"] .stSlider { margin-bottom: 8px; }

    /* Tabs */
    button[data-baseweb="tab"] { font-weight: 600; }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Market Parameters")

    # Preset scenarios
    preset = st.selectbox("Load Scenario", ["Custom", "Bull Market", "Bear Market", "Stable Market", "Volatile Mix"])

    if preset == "Bull Market":
        def_pp, def_pc, def_cp, def_cc, def_sp = 7200, 6.5, 110.0, 4.0, 27.0
    elif preset == "Bear Market":
        def_pp, def_pc, def_cp, def_cc, def_sp = 3500, -6.0, 65.0, -6.0, 17.5
    elif preset == "Stable Market":
        def_pp, def_pc, def_cp, def_cc, def_sp = 4500, 0.0, 82.0, 0.0, 21.0
    elif preset == "Volatile Mix":
        def_pp, def_pc, def_cp, def_cc, def_sp = 6000, 5.0, 95.0, -3.5, 24.0
    else:
        def_pp, def_pc, def_cp, def_cc, def_sp = 4000, 0.5, 80.0, 0.2, 20.0

    st.markdown("---")
    st.markdown("**Palm Oil Futures**")
    palm_price = st.slider("Price (MYR/MT)", 3000, 8500, def_pp, 50)
    palm_change = st.slider("Change (%)", -11.0, 12.0, float(def_pc), 0.1)

    st.markdown("**Crude Oil (WTI)**")
    crude_price = st.slider("Price (USD/bbl)", 60.0, 125.0, float(def_cp), 1.0)
    crude_change = st.slider("Change (%)", -13.0, 10.0, float(def_cc), 0.1)

    st.markdown("**US Sugar #11**")
    sugar_price = st.slider("Price (USD ¢/lb)", 16.0, 30.0, float(def_sp), 0.5)

    st.markdown("---")
    st.markdown("**Group:** Louis · Dimas · Gavin")
    st.markdown("**Course:** DKA Major Assignment")

# ==========================================
# RUN INFERENCE
# ==========================================
mamdani_res, aggregated, mu_mamdani, rule_str_mamdani = predict_mamdani(
    palm_price, palm_change, crude_price, crude_change, sugar_price)
sugeno_res, mu_sugeno, rule_str_sugeno = predict_sugeno(
    palm_price, palm_change, crude_price, crude_change, sugar_price)

price_diff = sugeno_res - mamdani_res
active_rules = sum(1 for s in rule_str_mamdani if s > 0)

# ==========================================
# HERO BANNER
# ==========================================
st.markdown(f"""
<div class="hero-banner">
    <h1>FuzzyOil</h1>
    <p>Cooking Oil Price Prediction via Fuzzy Logic Inference &nbsp;·&nbsp; Mamdani vs Sugeno</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Visualizations", "📋 Rule Base", "ℹ️ Methodology"])

# ========== TAB 1: DASHBOARD ==========
with tab1:
    # Top metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Mamdani (CoG)</div>
            <div class="value">Rp {mamdani_res:,.0f}</div>
            <div class="sub">Center of Gravity</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        delta_sign = "▲" if price_diff >= 0 else "▼"
        delta_color = "#FF3B30" if price_diff >= 0 else "#34C759"
        st.markdown(f"""<div class="metric-card">
            <div class="label">Sugeno (WAvg)</div>
            <div class="value">Rp {sugeno_res:,.0f}</div>
            <div class="sub" style="color:{delta_color}">{delta_sign} {abs(price_diff):,.0f} vs Mamdani</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Active Rules</div>
            <div class="value">{active_rules}</div>
            <div class="sub">of {len(RULES)} total rules fired</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        avg_price = (mamdani_res + sugeno_res) / 2
        st.markdown(f"""<div class="metric-card">
            <div class="label">Consensus Price</div>
            <div class="value">Rp {avg_price:,.0f}</div>
            <div class="sub">Mean of both methods</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison bar chart + fuzzification breakdown
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-header">Method Comparison</div>', unsafe_allow_html=True)
        chart_df = pd.DataFrame({
            "Method": ["Mamdani", "Sugeno"],
            "Price (IDR/liter)": [mamdani_res, sugeno_res]
        }).set_index("Method")
        st.bar_chart(chart_df, color=["#007AFF"])

        # MAE placeholder using synthetic ground truth for demo
        # This is a placeholder; replace with actual dataset evaluation
        st.markdown('<div class="section-header">Performance Metrics (Sample)</div>', unsafe_allow_html=True)
        np.random.seed(42)
        n_samples = 50
        sample_inputs = {
            'palm': np.random.uniform(3000, 8500, n_samples),
            'palm_c': np.random.uniform(-11, 12, n_samples),
            'crude': np.random.uniform(60, 125, n_samples),
            'crude_c': np.random.uniform(-13, 10, n_samples),
            'sugar': np.random.uniform(16, 30, n_samples),
        }
        mamdani_preds = [predict_mamdani(sample_inputs['palm'][i], sample_inputs['palm_c'][i],
                                          sample_inputs['crude'][i], sample_inputs['crude_c'][i],
                                          sample_inputs['sugar'][i])[0] for i in range(n_samples)]
        sugeno_preds  = [predict_sugeno(sample_inputs['palm'][i], sample_inputs['palm_c'][i],
                                         sample_inputs['crude'][i], sample_inputs['crude_c'][i],
                                         sample_inputs['sugar'][i])[0] for i in range(n_samples)]
        # Use Sugeno as pseudo ground truth for inter-method MAE illustration
        mae_mamdani = np.mean(np.abs(np.array(mamdani_preds) - np.array(sugeno_preds)))
        std_spread  = np.std(np.abs(np.array(mamdani_preds) - np.array(sugeno_preds)))

        m1, m2 = st.columns(2)
        m1.metric("Inter-method MAE", f"Rp {mae_mamdani:,.0f}", help="Mean absolute difference between Mamdani and Sugeno over 50 random samples")
        m2.metric("Spread (Std Dev)", f"Rp {std_spread:,.0f}", help="Standard deviation of method difference")

    with col_right:
        st.markdown('<div class="section-header">Fuzzification — Membership Degrees</div>', unsafe_allow_html=True)
        mu = mu_mamdani
        fuzz_rows = []
        for var, lbls in [
            ("Palm Price", mu['Palm_Price']),
            ("Palm Δ%",    mu['Palm_Change']),
            ("Crude Price",mu['Crude_Price']),
            ("Crude Δ%",   mu['Crude_Change']),
            ("Sugar Price",mu['Sugar_Price']),
        ]:
            for lbl, val in lbls.items():
                fuzz_rows.append({"Variable": var, "Label": lbl, "μ": round(val, 4)})

        fuzz_df = pd.DataFrame(fuzz_rows)
        # Style: only show non-zero rows if many zeros
        active_fuzz = fuzz_df[fuzz_df["μ"] > 0.001]
        display_df = active_fuzz if len(active_fuzz) > 0 else fuzz_df

        def highlight_mu(val):
            if val > 0.6:   return 'background-color: #E1F0FF; color: #007AFF; font-weight:700'
            elif val > 0.2: return 'background-color: #F0EEFF; color: #5856D6;'
            return ''

        styled = display_df.style.map(highlight_mu, subset=['μ']).format({'μ': '{:.4f}'})
        st.dataframe(styled, use_container_width=True, hide_index=True)

        if len(active_fuzz) < len(fuzz_df):
            st.caption(f"Showing {len(active_fuzz)} active memberships (μ > 0.001) of {len(fuzz_df)} total")

# ========== TAB 2: VISUALIZATIONS ==========
with tab2:
    st.markdown('<div class="section-header">Input Membership Functions</div>', unsafe_allow_html=True)
    st.caption("Dashed line = current input value · Shaded area = active membership degree")

    INPUT_RANGES = {
        'Palm_Price':  np.linspace(3000, 8500, 400),
        'Palm_Change': np.linspace(-11, 12, 400),
        'Crude_Price': np.linspace(60, 125, 400),
        'Crude_Change':np.linspace(-13, 10, 400),
        'Sugar_Price': np.linspace(16, 30, 400),
    }
    INPUT_VALUES = {
        'Palm_Price': palm_price, 'Palm_Change': palm_change,
        'Crude_Price': crude_price, 'Crude_Change': crude_change,
        'Sugar_Price': sugar_price,
    }
    NICE_NAMES = {
        'Palm_Price': 'Palm Oil Futures (MYR/MT)',
        'Palm_Change': 'Palm Oil Change (%)',
        'Crude_Price': 'Crude Oil WTI (USD/bbl)',
        'Crude_Change': 'Crude Oil Change (%)',
        'Sugar_Price': 'Sugar #11 (USD ¢/lb)',
    }

    col_a, col_b = st.columns(2)
    vars_list = list(INPUT_MFS.keys())
    for i, var in enumerate(vars_list):
        col = col_a if i % 2 == 0 else col_b
        with col:
            fig = plot_membership_functions(
                NICE_NAMES[var], INPUT_MFS[var],
                INPUT_VALUES[var], INPUT_RANGES[var]
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown('<div class="section-header">Mamdani — Output Aggregation</div>', unsafe_allow_html=True)
    fig_out = plot_output_aggregation(aggregated, mamdani_res)
    st.pyplot(fig_out, use_container_width=True)
    plt.close(fig_out)

    st.markdown('<div class="section-header">Active Rule Firing Strengths</div>', unsafe_allow_html=True)
    fig_rules = plot_rule_activations(rule_str_mamdani)
    if fig_rules:
        st.pyplot(fig_rules, use_container_width=True)
        plt.close(fig_rules)
    else:
        st.warning("No rules fired for the current inputs. Try adjusting the sliders.")

# ========== TAB 3: RULE BASE ==========
with tab3:
    st.markdown('<div class="section-header">Fuzzy Rule Base (20 Rules)</div>', unsafe_allow_html=True)

    rules_df = pd.DataFrame(RULES, columns=[
        "Palm Price", "Palm Δ%", "Crude Price", "Crude Δ%", "Sugar Price", "→ Output"
    ])
    rules_df.insert(0, "#", range(1, len(RULES)+1))
    rules_df["Firing Strength"] = [round(s, 4) for s in rule_str_mamdani]

    def highlight_rule(row):
        if row["Firing Strength"] > 0:
            return ['background-color: #E1F0FF'] * len(row)
        return [''] * len(row)

    def color_output(val):
        colors = {
            'Sangat_Rendah': 'color: #5856D6; font-weight:700',
            'Rendah':        'color: #34C759; font-weight:700',
            'Sedang':        'color: #007AFF; font-weight:700',
            'Tinggi':        'color: #FF9500; font-weight:700',
            'Sangat_Tinggi': 'color: #FF3B30; font-weight:700',
        }
        return colors.get(val, '')

    styled_rules = rules_df.style\
        .apply(highlight_rule, axis=1)\
        .map(color_output, subset=['→ Output'])\
        .format({'Firing Strength': '{:.4f}'})

    st.dataframe(styled_rules, use_container_width=True, hide_index=True, height=600)
    st.caption(f"🔵 Highlighted rows = currently active rules ({active_rules} of {len(RULES)} fired)")

# ========== TAB 4: METHODOLOGY ==========
with tab4:
    st.markdown('<div class="section-header">System Architecture</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Mamdani Method")
        st.markdown("""
**Fuzzification** → Convert crisp inputs into membership degrees using triangular (trimf) and trapezoidal (trapmf) functions.

**Inference** → Apply AND operator (min) across antecedents; each rule produces a clipped output MF.

**Aggregation** → Combine all clipped output MFs using OR (max).

**Defuzzification** → Compute the **Center of Gravity (CoG)** of the aggregated area.

$$z^* = \\frac{\\int z \\cdot \\mu_A(z)\\, dz}{\\int \\mu_A(z)\\, dz}$$

✅ **Pros:** Intuitive, linguistically interpretable  
❌ **Cons:** Computationally heavier (requires numeric integration)
""")

    with c2:
        st.markdown("#### Sugeno Method")
        st.markdown("""
**Fuzzification** → Same MFs and linguistic variables as Mamdani.

**Inference** → Same AND (min) operator for antecedents.

**Aggregation** → Output is a **singleton** constant per rule (no MF shapes needed).

**Defuzzification** → **Weighted Average** of singleton values.

$$z^* = \\frac{\\sum_i w_i \\cdot z_i}{\\sum_i w_i}$$

✅ **Pros:** Computationally efficient, lower MAE for numerical prediction  
❌ **Cons:** Less interpretable, singletons require calibration
""")

    st.markdown('<div class="section-header">Linguistic Variables Summary</div>', unsafe_allow_html=True)
    summary_data = {
        "Variable": ["Palm Oil Price (MYR/MT)", "Palm Oil Change (%)", "Crude Oil Price (USD/bbl)", "Crude Oil Change (%)", "Sugar #11 Price (USD ¢/lb)"],
        "Labels": ["Rendah · Sedang · Tinggi", "Turun · Stabil · Naik", "Rendah · Sedang · Tinggi", "Turun · Stabil · Naik", "Rendah · Sedang · Tinggi"],
        "Range": ["3,000 – 8,500", "-11 – +12", "60 – 125", "-13 – +10", "16 – 30"],
        "MF Types": ["Trap + Tri + Trap", "Trap + Tri + Trap", "Trap + Tri + Trap", "Trap + Tri + Trap", "Trap + Tri + Trap"],
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    st.markdown("**Output variable:** Cooking Oil Price (IDR/liter) — 5 labels: Sangat Rendah · Rendah · Sedang · Tinggi · Sangat Tinggi · Range: 13,000 – 37,000")