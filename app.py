import streamlit as st
import numpy as np

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
    'Rendah': lambda x: trimf(x, 15500, 16800, 18200),
    'Sedang': lambda x: trimf(x, 17500, 19000, 21000),
    'Tinggi': lambda x: trimf(x, 19500, 22000, 25000),
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
    ('Tinggi', 'Naik', 'Tinggi', 'Naik', 'Tinggi', 'Sangat_Tinggi'),
    ('Sedang', 'Naik', 'Sedang', 'Naik', 'Sedang', 'Tinggi'),
    ('Sedang', 'Turun', 'Rendah', 'Turun', 'Rendah', 'Rendah'),
    ('Rendah', 'Turun', 'Rendah', 'Turun', 'Rendah', 'Sangat_Rendah'),
    ('Sedang', 'Naik', 'Sedang', 'Stabil', 'Sedang', 'Sedang'),
    ('Tinggi', 'Naik', 'Tinggi', 'Naik', 'Tinggi', 'Sangat_Tinggi'),
    ('Tinggi', 'Stabil', 'Sedang', 'Stabil', 'Tinggi', 'Sangat_Tinggi'),
    ('Rendah', 'Stabil', 'Rendah', 'Stabil', 'Rendah', 'Sangat_Rendah'),
    ('Sedang', 'Stabil', 'Tinggi', 'Stabil', 'Sedang', 'Tinggi'),
    ('Sedang', 'Stabil', 'Rendah', 'Stabil', 'Tinggi', 'Sedang'),
    ('Rendah', 'Turun', 'Rendah', 'Turun', 'Rendah', 'Sangat_Rendah'),
    ('Sedang', 'Stabil', 'Tinggi', 'Naik', 'Rendah', 'Tinggi'),
    ('Rendah', 'Stabil', 'Tinggi', 'Stabil', 'Sedang', 'Sedang'),
    ('Tinggi', 'Stabil', 'Rendah', 'Stabil', 'Sedang', 'Sedang'),
    ('Sedang', 'Naik', 'Sedang', 'Naik', 'Tinggi', 'Sangat_Tinggi'),
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
        'Palm_Price': fuzzify(palm_price, INPUT_MFS['Palm_Price']),
        'Palm_Change': fuzzify(palm_change, INPUT_MFS['Palm_Change']),
        'Crude_Price': fuzzify(crude_price, INPUT_MFS['Crude_Price']),
        'Crude_Change': fuzzify(crude_change, INPUT_MFS['Crude_Change']),
        'Sugar_Price': fuzzify(sugar_price, INPUT_MFS['Sugar_Price']),
    }
    aggregated = np.zeros(len(OUTPUT_RANGE))
    for p_lbl, pc_lbl, c_lbl, cc_lbl, s_lbl, out_lbl in RULES:
        fs = min(
            mu['Palm_Price'].get(p_lbl, 0), mu['Palm_Change'].get(pc_lbl, 0),
            mu['Crude_Price'].get(c_lbl, 0), mu['Crude_Change'].get(cc_lbl, 0),
            mu['Sugar_Price'].get(s_lbl, 0)
        )
        if fs > 0:
            clipped = np.minimum(fs, OUTPUT_MFS[out_lbl](OUTPUT_RANGE))
            aggregated = np.maximum(aggregated, clipped)
    return defuzzify_centroid(aggregated, OUTPUT_RANGE)

def predict_sugeno(palm_price, palm_change, crude_price, crude_change, sugar_price):
    mu = {
        'Palm_Price': fuzzify(palm_price, INPUT_MFS['Palm_Price']),
        'Palm_Change': fuzzify(palm_change, INPUT_MFS['Palm_Change']),
        'Crude_Price': fuzzify(crude_price, INPUT_MFS['Crude_Price']),
        'Crude_Change': fuzzify(crude_change, INPUT_MFS['Crude_Change']),
        'Sugar_Price': fuzzify(sugar_price, INPUT_MFS['Sugar_Price']),
    }
    numerator, denominator = 0.0, 0.0
    for p_lbl, pc_lbl, c_lbl, cc_lbl, s_lbl, out_lbl in RULES:
        fs = min(
            mu['Palm_Price'].get(p_lbl, 0), mu['Palm_Change'].get(pc_lbl, 0),
            mu['Crude_Price'].get(c_lbl, 0), mu['Crude_Change'].get(cc_lbl, 0),
            mu['Sugar_Price'].get(s_lbl, 0)
        )
        numerator += fs * OUTPUT_SINGLETONS[out_lbl]
        denominator += fs
    return numerator / denominator if denominator > 1e-10 else np.mean(list(OUTPUT_SINGLETONS.values()))

# ==========================================
# 4. STREAMLIT UI (UPGRADED DASHBOARD)
# ==========================================
import pandas as pd # Make sure to import pandas for the chart!

st.set_page_config(page_title="Fuzzy Cooking Oil Price", page_icon="🛢️", layout="centered")

st.title("Cooking Oil Price Prediction")
st.markdown("**Method:** Fuzzy Logic Inference (Mamdani vs Sugeno)")
st.markdown("**Group:** Louis, Dimas, Gavin | DKA Major Assignment")
st.divider()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Market Parameter Inputs")

palm_price = st.sidebar.slider("Palm Oil Futures (MYR/MT)", min_value=3000, max_value=8500, value=4000, step=50)
palm_change = st.sidebar.slider("Palm Oil Change (%)", min_value=-11.0, max_value=12.0, value=0.5, step=0.1)
crude_price = st.sidebar.slider("Crude Oil WTI (USD/bbl)", min_value=60.0, max_value=125.0, value=80.0, step=1.0)
crude_change = st.sidebar.slider("Crude Oil Change (%)", min_value=-13.0, max_value=10.0, value=0.2, step=0.1)
sugar_price = st.sidebar.slider("US Sugar #11 (USD cents/lb)", min_value=16.0, max_value=30.0, value=20.0, step=0.5)

# --- CALCULATIONS ---
# We calculate these first so we can use them in both the metrics and the chart
mamdani_res = predict_mamdani(palm_price, palm_change, crude_price, crude_change, sugar_price)
sugeno_res = predict_sugeno(palm_price, palm_change, crude_price, crude_change, sugar_price)
price_difference = sugeno_res - mamdani_res

# --- DASHBOARD UI ---
st.write("### 📊 Prediction Results")

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Mamdani (Center of Gravity)", value=f"IDR {mamdani_res:,.0f}")
    st.caption("Best for representing abstract concepts and linguistic interpretation.")

with col2:
    st.metric(
        label="Sugeno (Weighted Average)", 
        value=f"IDR {sugeno_res:,.0f}", 
        delta=f"{price_difference:,.0f} vs Mamdani",
        delta_color="inverse" # Makes a lower price green (good for consumers), higher price red
    )
    st.caption("More accurate for mathematical regression with a lower MAE.")

# --- VISUAL CHART ---
st.write("### Price Comparison Chart")
chart_data = pd.DataFrame(
    {"Method": ["Mamdani", "Sugeno"], "Price (IDR)": [mamdani_res, sugeno_res]}
)
st.bar_chart(chart_data.set_index("Method"), color="#E63946") # Using a nice red color

st.divider()

# --- COLLAPSIBLE INFO SECTIONS ---
with st.expander("View 20 Fuzzy Rules (Rule Base)"):
    st.write("The system evaluates the following rules instantly behind the scenes:")
    # Create a nice dataframe to display the rules clearly
    rules_df = pd.DataFrame(RULES, columns=["Palm Price", "Palm %", "Crude Price", "Crude %", "Sugar Price", "Output Result"])
    st.dataframe(rules_df, use_container_width=True)

with st.expander("Methodology Explanation"):
    st.markdown("""
    * **Mamdani Method:** Uses geometric area integration (Center of Gravity) to find the balance point of overlapping fuzzy concepts. Highly interpretable but computationally heavy.
    * **Sugeno Method:** Uses a weighted average of constant values (Singletons). Bypasses geometric shapes entirely, making it much faster and more precise for continuous numerical tracking.
    """)