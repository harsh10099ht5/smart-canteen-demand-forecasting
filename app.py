
import streamlit as st
import pandas as pd
import numpy as np

from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Canteen AI",
    page_icon="🍽️",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #f7f8fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .brand {
        font-size: 30px;
        font-weight: 800;
        color: #171923;
    }

    .subtitle {
        color: #6b7280;
        font-size: 14px;
    }

    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #e6e8ef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }

    .metric-title {
        color: #6b7280;
        font-size: 13px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #171923;
    }

    .prediction-box {
        background: #eef4ff;
        border: 1px solid #c9dcff;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
    }

    .prediction-number {
        font-size: 48px;
        font-weight: 800;
        color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD AND CLEAN DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "canteen_data.csv",
        parse_dates=["date"]
    )

    # Fill missing temperature values
    df["temperature"] = df["temperature"].fillna(
        df["temperature"].median()
    )

    # Convert invalid negative sales to positive values
    df["units_sold"] = df["units_sold"].abs()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Feature engineering
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Sort before calculating rolling demand
    df = df.sort_values(["item", "date"])

    df["rolling_avg_7"] = (
        df.groupby("item")["units_sold"]
        .transform(
            lambda s: s.shift(1)
            .rolling(7, min_periods=1)
            .mean()
        )
    )

    # Fill first few rolling values
    df["rolling_avg_7"] = df["rolling_avg_7"].fillna(
        df.groupby("item")["units_sold"]
        .transform("mean")
    )

    return df.reset_index(drop=True)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

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

    return (
        model,
        scaler,
        feature_columns,
        metrics
    )


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

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

    # Match the exact columns used during training
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


# --------------------------------------------------
# APPLICATION START
# --------------------------------------------------

try:
    df = load_data()

    model, scaler, feature_columns, metrics = train_model(df)

except Exception as error:
    st.error(f"Unable to load or train the model: {error}")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🍽️ Canteen AI")

    st.caption("Smart demand intelligence")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Predict Demand",
            "Dataset",
            "Model Performance"
        ]
    )

    st.divider()

    st.caption("Prototype v1.0")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="brand">Canteen AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Smart food demand prediction and inventory planning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    st.subheader("Dashboard Overview")

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
                <div class="metric-title">Total units sold</div>
                <div class="metric-value">{total_units:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Average units</div>
                <div class="metric-value">{average_units:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Top-selling item</div>
                <div class="metric-value">{top_item}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Dataset records</div>
                <div class="metric-value">{total_records:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    left, right = st.columns(2)

    with left:

        st.subheader("Demand by Item")

        item_summary = (
            df.groupby("item")["units_sold"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(item_summary)

    with right:

        st.subheader("Average Demand Signals")

        signals = pd.DataFrame({
            "Condition": [
                "Weekday",
                "Weekend",
                "Exam Day",
                "Festival Day"
            ],
            "Average Units": [
                df.loc[~df["is_weekend"], "units_sold"].mean(),
                df.loc[df["is_weekend"], "units_sold"].mean(),
                df.loc[df["is_exam"], "units_sold"].mean(),
                df.loc[df["is_festival"], "units_sold"].mean()
            ]
        })

        st.bar_chart(
            signals.set_index("Condition")
        )

    st.subheader("Recent Sales Records")

    st.dataframe(
        df.sort_values("date", ascending=False).head(10),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

elif page == "Predict Demand":

    st.subheader("Predict Food Demand")

    st.caption(
        "Enter the expected conditions for the selected day."
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

        preparation_quantity = int(np.ceil(prediction * 1.08))

        st.markdown(
            f"""
            <div class="prediction-box">
                <div>Estimated demand for {item}</div>
                <div class="prediction-number">
                    {prediction}
                </div>
                <div>units</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Predicted demand",
                f"{prediction} units"
            )

        with col2:
            st.metric(
                "Suggested preparation",
                f"{preparation_quantity} units"
            )

        with col3:
            st.metric(
                "Safety buffer",
                "8%"
            )

        st.info(
            "This is a machine-learning estimate, not a guaranteed "
            "sales figure. The preparation quantity includes an "
            "8% planning buffer."
        )


# --------------------------------------------------
# DATASET PAGE
# --------------------------------------------------

elif page == "Dataset":

    st.subheader("Dataset Explorer")

    st.caption(
        "Explore and filter the canteen sales data."
    )

    selected_item = st.selectbox(
        "Filter by item",
        ["All"] + sorted(df["item"].unique())
    )

    filtered_df = df.copy()

    if selected_item != "All":
        filtered_df = filtered_df[
            filtered_df["item"] == selected_item
        ]

    st.write(
        f"Showing {len(filtered_df):,} records"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download filtered CSV",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_canteen_data.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------

elif page == "Model Performance":

    st.subheader("Model Performance")

    st.caption(
        "Evaluation on the held-out test set."
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

    st.markdown("### Interpretation")

    st.write(
        "MAE represents the average absolute prediction error. "
        "RMSE gives more weight to larger errors. "
        "R² indicates how much variation in the target is explained "
        "by the model on the test split."
    )

    st.warning(
        "The current notebook uses a random train-test split. "
        "For reliable future-day forecasting, evaluate using a "
        "chronological time-based split."
    )