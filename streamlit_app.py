import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Supermarket Rating Predictor", page_icon="🛒")

st.title("🛒 Supermarket Customer Rating Predictor")
st.write("Fill in the transaction details below to predict the customer rating.")

# ── Load pipeline steps separately ────────────────────────────────────────────
@st.cache_resource
def load_steps():
    pipeline = joblib.load("pipeline.pkl")
    scaler = pipeline.named_steps["scaler"]
    model  = pipeline.named_steps["model"]
    return scaler, model

try:
    scaler, model = load_steps()
except FileNotFoundError:
    st.error("pipeline.pkl not found. Make sure it is in the same folder as app.py.")
    st.stop()

# ── Product line target-encoded means (from training data) ─────────────────────
PRODUCT_LINE_ENCODING = {
    "Health and beauty":       6.84,
    "Electronic accessories":  7.03,
    "Home and lifestyle":      6.93,
    "Sports and travel":       7.07,
    "Food and beverages":      7.11,
    "Fashion accessories":     7.00,
}

# ── Input form ─────────────────────────────────────────────────────────────────
with st.form("predict_form"):
    st.subheader("Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        unit_cost    = col1.number_input("Unit Price ($)", min_value=0.0, value=55.0, step=0.5)
        quantity     = col1.number_input("Quantity", min_value=1, value=3, step=1)
        revenue      = unit_cost * quantity
        st.caption(f"Revenue (auto): **${revenue:.2f}**")
        branch       = col1.selectbox("Branch", ["A", "B", "C"])
        product_line = col1.selectbox("Product Line", list(PRODUCT_LINE_ENCODING.keys()))

    with col2:
        customer_type = col2.selectbox("Customer Type", ["Member", "Normal"])
        gender        = col2.selectbox("Gender", ["Male", "Female"])
        payment       = col2.selectbox("Payment Method", ["Cash", "Credit card", "Ewallet"])
        hour          = col2.slider("Hour of Purchase", min_value=10, max_value=20, value=14)
        weekday       = col2.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )

    submitted = st.form_submit_button("Predict Rating", use_container_width=True)

# ── Build feature row & predict ────────────────────────────────────────────────
if submitted:
    weekday_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    input_df = pd.DataFrame([{
        "customer_type":              1 if customer_type == "Member" else 0,
        "gender_customer":            1 if gender == "Male" else 0,
        "product_line":               PRODUCT_LINE_ENCODING[product_line],
        "unit_cost":                  unit_cost,
        "quantity":                   float(quantity),
        "revenue":                    revenue,
        "hour":                       float(hour),
        "weekday":                    float(weekday_map[weekday]),
        "branch_B":                   1 if branch == "B" else 0,
        "branch_C":                   1 if branch == "C" else 0,
        "payment_method_Credit card": 1 if payment == "Credit card" else 0,
        "payment_method_Ewallet":     1 if payment == "Ewallet" else 0,
    }])

    # Apply scaler only to the 5 columns it was trained on
    NUM_COLS = ["unit_cost", "revenue", "quantity", "hour", "weekday"]
    input_df[NUM_COLS] = scaler.transform(input_df[NUM_COLS])

    prediction = float(model.predict(input_df)[0])
    prediction = max(4.0, min(10.0, prediction))

    st.divider()
    st.metric(label="Predicted Customer Rating", value=f"{prediction:.2f} / 10")
