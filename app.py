
import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Canteen AI | Smart Forecasting",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# PROFESSIONAL UI - INLINE CSS
# ==================================================

st.markdown("""
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f7fb;
}

.block-container {
    max-width: 1450px;
    padding: 2rem 3rem 3rem 3rem;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: #101828;
    border-right: 1px solid #1d2939;
}

[data-testid="stSidebar"] * {
    color: #f9fafb;
}

[data-testid="stSidebar"] .stRadio label {
    padding: 9px 12px;
    border-radius: 8px;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: #1d2939;
}

/* Headings */

h1, h2, h3 {
    color: #101828;
    font-weight: 700;
    letter-spacing: -0.5px;
}

p {
    color: #667085;
}

/* Header */

.hero {
    background: linear-gradient(
        115deg,
        #172554 0%,
        #1d4ed8 100%
    );
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(29, 78, 216, 0.12);
}

.hero h1 {
    color: #ffffff;
    font-size: 34px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #dbeafe;
    font-size: 15px;
    margin-top: 8px;
}

/* Metric cards */

.metric-card {
    background: #ffffff;
    border: 1px solid #eaecf0;
    border-radius: 16px;
    padding: 22px;
    min-height: 140px;
    box-shadow: 0 4px 12px rgba(16, 24, 40, 0.04);
}

.metric-title {
    color: #667085;
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 12px;
}

.metric-value {
    color: #101828;
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -1px;
    word-break: break-word;
}

/* Section heading */

.section-title {
    color: #101828;
    font-size: 23px;
    font-weight: 700;
    margin-top: 22px;
    margin-bottom: 5px;
}

/* Cards */

.content-card {
    background: #ffffff;
    border: 1px solid #eaecf0;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}

/* Prediction result */

.prediction-box {
    background: linear-gradient(
        135deg,
        #eff6ff,
        #dbeafe
    );
    border: 1px solid #bfdbfe;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
}

.prediction-label {
    color: #1e40af;
    font-size: 15px;
    font-weight: 600;
}

.prediction-number {
    color: #1d4ed8;
    font-size: 58px;
    font-weight: 800;
    margin: 8px 0;
}

.prediction-unit {
    color: #1e40af;
    font-size: 14px;
}

/* Buttons */

.stButton > button {
    border-radius: 10px;
    min-height: 45px;
    font-weight: 600;
}

/* Input controls */

[data-testid="stNumberInput"],
[data-testid="stDateInput"],
[data-testid="stSelectbox"] {
    border-radius: 10px;
}

/* Hide Streamlit branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Responsive spacing */

@media (max-width: 900px) {
    .block-container {
        padding: 1.5rem;
    }

    .hero h1 {
        font-size: 27px;
    }

    .metric-value {
        font-size: 24px;
    }
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# DATA LOADING
# ==================================================

@st.cache_data
def load_data():

    file_path = Path(__file__).parent / "canteen_data.csv"

    df = pd.read_csv(
        file_path,
        parse_dates=["date"]
    )

    # Clean missing temperature values
    df["temperature"] = df["temperature"].fillna(
        df["temperature"].median()
    )

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Keep sales values non-negative
    df["units_sold"] = df["units_sold"].abs()

    # Feature engineering
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Calculate previous demand for each food item
    df = df.sort_values(["item", "date"])

    df["rolling_avg_7"] = (
        df.groupby("item")["units_sold"]
        .transform(
            lambda s: s.shift(1)
            .rolling(7, min_periods=1)
            .mean()
        )
    )

    # Fill missing rolling averages
    df["rolling_avg_7"] = df["rolling_avg_7"].fillna(
        df.groupby("item")["units_sold"]
        .transform("mean")
    )

    return df.reset_index(drop=True)


# ==================================================
# MODEL TRAINING
# ==================================================

@st.cache_resource
def train_model(df):

    model_df = pd.get_dummies(
        df,
        columns=["item"],
        drop_first=True,
        dtype=int
    )

    feature_columns = [
        col for col in model_df.columns
        if col not in ["units_sold", "date"]
    ]

    X = model_df[feature_columns]
    y = model_df["units_sold"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    numeric_columns = [
        "temperature",
        "rainfall",
        "rolling_avg_7"
    ]

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numeric_columns] = scaler.fit_transform(
        X_train[numeric_columns]
    )

    X_test[numeric_columns] = scaler.transform(
        X_test[numeric_columns]
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "R2": r2_score(y_test, predictions)
    }

    return model, scaler, feature_columns, metrics


# ==================================================
# DEMAND PREDICTION
# ==================================================

def predict_demand(
    model,
    scaler,
    feature_columns,
    item,
    selected_date,
    temperature,
    rainfall,
    is_exam,
    is_festival,
    rolling_avg
):

    is_weekend = int(selected_date.weekday() >= 5)

    input_data = pd.DataFrame([{
        "is_weekend": is_weekend,
        "is_exam": int(is_exam),
        "is_festival": int(is_festival),
        "temperature": temperature,
        "rainfall": rainfall,
        "day_of_week": selected_date.weekday(),
        "month": selected_date.month,
        "rolling_avg_7": rolling_avg,
        "item": item
    }])

    input_data = pd.get_dummies(
        input_data,
        columns=["item"],
        drop_first=True,
        dtype=int
    )

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    numeric_columns = [
        "temperature",
        "rainfall",
        "rolling_avg_7"
    ]

    input_data[numeric_columns] = scaler.transform(
        input_data[numeric_columns]
    )

    prediction = model.predict(input_data)[0]

    return max(0, round(float(prediction)))


# ==================================================
# LOAD APPLICATION
# ==================================================

try:

    df = load_data()

    model, scaler, feature_columns, metrics = train_model(df)

except Exception as error:

    st.error(f"Application error: {error}")
    st.stop()


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:

    st.markdown("# 🍽️ Canteen AI")

    st.caption("Smart Demand Intelligence")

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Select module",
        [
            "Overview",
            "Predict Demand",
            "Dataset",
            "Model Performance"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### System Status")

    st.success("Model loaded")

    st.caption("Data source: canteen_data.csv")

    st.caption("Version 1.0 • Prototype")


# ==================================================
# MAIN HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <h1>🍽️ Canteen AI</h1>
    <p>
        Smart food demand forecasting and inventory intelligence
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# OVERVIEW PAGE
# ==================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">Dashboard Overview</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Monitor historical sales and understand demand patterns."
    )

    total_units = int(df["units_sold"].sum())
    average_units = df["units_sold"].mean()
    total_records = len(df)
    top_item = df.groupby("item")["units_sold"].sum().idxmax()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Units Sold</div>
                <div class="metric-value">{total_units:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Average Demand</div>
                <div class="metric-value">{average_units:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Top-Selling Item</div>
                <div class="metric-value">{top_item}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Dataset Records</div>
                <div class="metric-value">{total_records:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="section-title">Demand by Item</div>',
            unsafe_allow_html=True
        )

        item_summary = (
            df.groupby("item")["units_sold"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(
            item_summary,
            use_container_width=True
        )

    with right:

        st.markdown(
            '<div class="section-title">Demand by Condition</div>',
            unsafe_allow_html=True
        )

        signals = pd.DataFrame({
            "Condition": [
                "Weekday",
                "Weekend",
                "Exam Day",
                "Festival Day"
            ],
            "Average Units": [
                df.loc[
                    ~df["is_weekend"], "units_sold"
                ].mean(),
                df.loc[
                    df["is_weekend"], "units_sold"
                ].mean(),
                df.loc[
                    df["is_exam"], "units_sold"
                ].mean(),
                df.loc[
                    df["is_festival"], "units_sold"
                ].mean()
            ]
        })

        st.bar_chart(
            signals.set_index("Condition"),
            use_container_width=True
        )

    st.markdown(
        '<div class="section-title">Recent Sales Records</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.sort_values(
            "date",
            ascending=False
        ).head(10),
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# PREDICTION PAGE
# ==================================================

elif page == "Predict Demand":

    st.markdown(
        '<div class="section-title">Predict Food Demand</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Enter the expected conditions to estimate food demand."
    )

    left, right = st.columns(2)

    with left:

        st.markdown("### Day Information")

        selected_date = st.date_input(
            "Select date",
            value=date.today()
        )

        item = st.selectbox(
            "Select food item",
            sorted(df["item"].unique())
        )

        is_exam = st.checkbox("Exam day")

        is_festival = st.checkbox("Festival day")

    with right:

        st.markdown("### Environmental Factors")

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=28.0,
            step=0.5
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=2.0,
            step=0.5
        )

        item_history = (
            df[df["item"] == item]["units_sold"]
            .tail(7)
            .mean()
        )

        rolling_avg = st.number_input(
            "Recent average demand",
            min_value=0.0,
            value=float(item_history),
            step=1.0
        )

    st.divider()

    if st.button(
        "Generate Demand Prediction",
        type="primary",
        use_container_width=True
    ):

        prediction = predict_demand(
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            item=item,
            selected_date=selected_date,
            temperature=temperature,
            rainfall=rainfall,
            is_exam=is_exam,
            is_festival=is_festival,
            rolling_avg=rolling_avg
        )

        preparation_quantity = int(
            np.ceil(prediction * 1.08)
        )

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-label">
                    Estimated demand for {item}
                </div>

                <div class="prediction-number">
                    {prediction}
                </div>

                <div class="prediction-unit">
                    Units required
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Predicted Demand",
                f"{prediction} units"
            )

        with col2:
            st.metric(
                "Suggested Preparation",
                f"{preparation_quantity} units"
            )

        with col3:
            st.metric(
                "Safety Buffer",
                "8%"
            )

        st.info(
            "This is a machine-learning estimate, not a guaranteed "
            "sales figure. The preparation quantity includes an "
            "8% planning buffer."
        )


# ==================================================
# DATASET PAGE
# ==================================================

elif page == "Dataset":

    st.markdown(
        '<div class="section-title">Dataset Explorer</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Explore, filter, and download the canteen sales dataset."
    )

    selected_item = st.selectbox(
        "Filter by food item",
        ["All"] + sorted(df["item"].unique())
    )

    filtered_df = df.copy()

    if selected_item != "All":

        filtered_df = filtered_df[
            filtered_df["item"] == selected_item
        ]

    st.info(
        f"Showing {len(filtered_df):,} records"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Filtered CSV",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_canteen_data.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==================================================
# MODEL PERFORMANCE PAGE
# ==================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Evaluation results from the held-out test set."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "MAE",
            f"{metrics['MAE']:.2f}"
        )

    with col2:
        st.metric(
            "RMSE",
            f"{metrics['RMSE']:.2f}"
        )

    with col3:
        st.metric(
            "R² Score",
            f"{metrics['R2']:.4f}"
        )

    st.divider()

    st.markdown("### Metric Interpretation")

    st.markdown("""
    - **MAE:** Average absolute prediction error.
    - **RMSE:** Error metric that gives more weight to larger errors.
    - **R² Score:** Proportion of target variation explained by the model
      on the test split.
    """)

    st.warning(
        "The current model uses a random train-test split. "
        "For reliable future-day forecasting, a chronological "
        "time-based evaluation should be added."
    )