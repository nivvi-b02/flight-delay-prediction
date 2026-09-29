import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from io import BytesIO
import base64

# ============================================================
# OPTIONAL REPORTING LIBRARIES
# ============================================================

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from openpyxl import Workbook
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"
REPORTS_PATH = PROJECT_ROOT / "reports"

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "flights_features.csv"

# Possible locations for original airport/airline files
AIRPORT_PATHS = [
    PROJECT_ROOT / "data" / "airports.csv",
    PROJECT_ROOT / "data" / "raw" / "airports.csv",
    PROJECT_ROOT / "airports.csv"
]

AIRLINE_PATHS = [
    PROJECT_ROOT / "data" / "airlines.csv",
    PROJECT_ROOT / "data" / "raw" / "airlines.csv",
    PROJECT_ROOT / "airlines.csv"
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


# ============================================================
# THEME
# ============================================================

if st.session_state.dark_mode:

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #111111;
            color: #F5F5F5;
        }

        .main .block-container {
            color: #F5F5F5;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF !important;
        }

        p, span, label, div {
            color: inherit;
        }

        .stMarkdown,
        .stText,
        .stCaption {
            color: #E6E6E6 !important;
        }

        [data-testid="stSidebar"] {
            background-color: #1A1A1A;
        }

        [data-testid="stSidebar"] * {
            color: #F5F5F5 !important;
        }

        [data-testid="stMetric"] {
            background-color: #1E1E1E;
            border: 1px solid #333333;
            padding: 12px;
            border-radius: 10px;
        }

        [data-testid="stMetricLabel"] {
            color: #CCCCCC !important;
        }

        [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
        }

        [data-testid="stDataFrame"] {
            background-color: #1E1E1E;
        }

        .stDataFrame div {
            color: #FFFFFF !important;
        }

        input, textarea {
            color: #FFFFFF !important;
            background-color: #222222 !important;
        }

        [data-baseweb="select"] {
            background-color: #222222 !important;
        }

        [data-baseweb="select"] * {
            color: #FFFFFF !important;
        }

        .stSelectbox label,
        .stNumberInput label,
        .stTextInput label,
        .stMultiSelect label {
            color: #FFFFFF !important;
        }

        .stAlert {
            color: #FFFFFF !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #F7F9FC;
            color: #202124;
        }

        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #202124 !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# LOAD MAIN DATASET
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD AIRPORT DATA
# ============================================================

@st.cache_data
def load_airports():

    for path in AIRPORT_PATHS:

        if path.exists():

            df = pd.read_csv(path)

            return df

    return pd.DataFrame()


# ============================================================
# LOAD AIRLINE DATA
# ============================================================

@st.cache_data
def load_airlines():

    for path in AIRLINE_PATHS:

        if path.exists():

            df = pd.read_csv(path)

            return df

    return pd.DataFrame()


# ============================================================
# GENERIC CSV LOADER
# ============================================================

@st.cache_data
def load_report_csv(filename):

    path = REPORTS_PATH / filename

    if path.exists():

        return pd.read_csv(path)

    return pd.DataFrame()


# ============================================================
# HELPER: FIND COLUMN
# ============================================================

def find_column(df, possible_names):

    if df.empty:
        return None

    normalized = {
        str(col).upper().replace(" ", "_"): col
        for col in df.columns
    }

    for name in possible_names:

        key = name.upper().replace(" ", "_")

        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def get_basic_statistics():

    df = load_data()

    stats = {}

    stats["total_flights"] = len(df)
    stats["features"] = len(df.columns) - 1

    if "AIRLINE" in df.columns:
        stats["airlines"] = df["AIRLINE"].nunique()
    else:
        stats["airlines"] = 0

    airports = set()

    if "ORIGIN_AIRPORT" in df.columns:
        airports.update(df["ORIGIN_AIRPORT"].dropna().astype(str).unique())

    if "DESTINATION_AIRPORT" in df.columns:
        airports.update(df["DESTINATION_AIRPORT"].dropna().astype(str).unique())

    stats["airports"] = len(airports)

    if "IS_DELAYED" in df.columns:

        stats["delayed"] = int(df["IS_DELAYED"].sum())
        stats["not_delayed"] = int((df["IS_DELAYED"] == 0).sum())
        stats["delay_rate"] = df["IS_DELAYED"].mean()

    else:

        stats["delayed"] = 0
        stats["not_delayed"] = 0
        stats["delay_rate"] = 0

    if "SCHEDULED_TIME" in df.columns:
        stats["avg_scheduled_time"] = df["SCHEDULED_TIME"].mean()

    if "DISTANCE" in df.columns:
        stats["avg_distance"] = df["DISTANCE"].mean()

    if "ROUTE" in df.columns:
        stats["routes"] = df["ROUTE"].nunique()

    elif "ORIGIN_AIRPORT" in df.columns and "DESTINATION_AIRPORT" in df.columns:

        stats["routes"] = (
            df["ORIGIN_AIRPORT"].astype(str)
            + "_"
            + df["DESTINATION_AIRPORT"].astype(str)
        ).nunique()

    else:
        stats["routes"] = 0

    return stats


stats = get_basic_statistics()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("✈️ Flight Analytics")
st.sidebar.markdown("### Flight Delay Prediction")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Introduction",
        "📈 Statistics",
        "📊 EDA",
        "🗺️ Airports & Routes",
        "✈️ Prediction",
        "🤖 Model Performance",
        "📄 Reports & Export",
        "📖 Glossary",
        "⚙️ Settings"
    ]
)


# ============================================================
# INTRODUCTION
# ============================================================

if page == "🏠 Introduction":

    st.title("✈️ Flight Delay Prediction")

    st.subheader(
        "Predicting flight delays using machine learning"
    )

    st.markdown(
        """
        This project uses historical flight data to predict whether
        a scheduled flight is likely to be delayed by more than
        15 minutes.

        The model uses information available before departure,
        including airline, origin airport, destination airport,
        scheduled departure time, scheduled duration and distance.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # DATASET STATISTICS
    # --------------------------------------------------------

    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Flights",
        f"{stats['total_flights']:,}"
    )

    col2.metric(
        "Features",
        stats["features"]
    )

    col3.metric(
        "Airlines",
        stats["airlines"]
    )

    col4.metric(
        "Airports",
        stats["airports"]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Delayed Flights",
        f"{stats['delayed']:,}"
    )

    col2.metric(
        "Delay Rate",
        f"{stats['delay_rate']:.2%}"
    )

    col3.metric(
        "Unique Routes",
        f"{stats['routes']:,}"
    )

    col4.metric(
        "Avg Distance",
        f"{stats.get('avg_distance', 0):,.1f} mi"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    st.subheader("🎯 Prediction Target")

    st.info(
        """
        **IS_DELAYED**

        **0 → Not delayed**

        **1 → Delayed by more than 15 minutes**
        """
    )

    # --------------------------------------------------------
    # DATASET SPLIT
    # --------------------------------------------------------

    st.subheader("🔀 Dataset Split")

    split_df = pd.DataFrame(
        {
            "Dataset": [
                "Training Set",
                "Test Set",
                "Total"
            ],
            "Flights": [
                4_571_206,
                1_142_802,
                5_714_008
            ],
            "Percentage": [
                "80%",
                "20%",
                "100%"
            ]
        }
    )

    st.dataframe(
        split_df,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        """
        The data was divided using an **80/20 train-test split**.
        Stratified sampling was used so that the proportion of
        delayed and non-delayed flights was preserved in both sets.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # ML PIPELINE
    # --------------------------------------------------------

    st.subheader("🔄 ML Pipeline")

    pipeline_status = pd.DataFrame(
        {
            "Pipeline Stage": [
                "Data Ingestion",
                "Data Profiling",
                "Data Understanding",
                "Schema Definition",
                "Data Validation",
                "Preprocessing",
                "Train-Test Split",
                "Exploratory Data Analysis",
                "Feature Engineering",
                "Model Training",
                "Model Evaluation",
                "Best Model Selection",
                "Prediction",
                "Streamlit Web App"
            ],
            "Status": [
                "✅ Completed",
                "✅ Completed",
                "✅ Completed",
                "✅ Completed",
                "✅ Completed",
                "✅ Completed",
                "✅ Completed",
                "✅ Completed",
                "✅ Completed",
                "✅ Completed",
                "✅ Completed",
                "✅ Completed",
                "✅ Completed",
                "✅ Completed"
            ]
        }
    )

    st.dataframe(
        pipeline_status,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # MODELS USED
    # --------------------------------------------------------

    st.subheader("🤖 Models Used")

    models_df = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "Linear SVM"
            ],
            "Type": [
                "Linear Classification",
                "Ensemble Tree Model",
                "Linear Classification"
            ]
        }
    )

    st.dataframe(
        models_df,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# STATISTICS
# ============================================================

elif page == "📈 Statistics":

    st.title("📈 Dataset Statistics")

    st.write(
        "A quick statistical overview of the flight dataset."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # BASIC NUMBERS
    # --------------------------------------------------------

    st.subheader("Dataset Size")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Flights",
        f"{stats['total_flights']:,}"
    )

    c2.metric(
        "Delayed",
        f"{stats['delayed']:,}"
    )

    c3.metric(
        "Not Delayed",
        f"{stats['not_delayed']:,}"
    )

    c4.metric(
        "Delay Rate",
        f"{stats['delay_rate']:.2%}"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # FLIGHT CHARACTERISTICS
    # --------------------------------------------------------

    st.subheader("✈️ Flight Characteristics")

    characteristics = pd.DataFrame(
        {
            "Statistic": [
                "Total Flights",
                "Airlines",
                "Airports",
                "Unique Routes",
                "Average Scheduled Flight Time",
                "Average Distance",
                "Delayed Flights",
                "Non-Delayed Flights",
                "Overall Delay Rate"
            ],
            "Value": [
                f"{stats['total_flights']:,}",
                f"{stats['airlines']:,}",
                f"{stats['airports']:,}",
                f"{stats['routes']:,}",
                f"{stats.get('avg_scheduled_time', 0):,.2f} minutes",
                f"{stats.get('avg_distance', 0):,.2f} miles",
                f"{stats['delayed']:,}",
                f"{stats['not_delayed']:,}",
                f"{stats['delay_rate']:.2%}"
            ]
        }
    )

    st.dataframe(
        characteristics,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("🎯 Target Distribution")

    target_df = pd.DataFrame(
        {
            "Class": [
                "Not Delayed",
                "Delayed"
            ],
            "Flights": [
                stats["not_delayed"],
                stats["delayed"]
            ],
            "Percentage": [
                stats["not_delayed"] / stats["total_flights"] * 100,
                stats["delayed"] / stats["total_flights"] * 100
            ]
        }
    )

    st.dataframe(
        target_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    st.subheader("🤖 Model Performance Summary")

    comparison = load_report_csv(
        "model_comparison.csv"
    )

    if not comparison.empty:

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# EDA
# ============================================================

elif page == "📊 EDA":

    st.title("📊 Exploratory Data Analysis")

    st.write(
        "Explore patterns and trends in historical flight delays."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("🎯 Overall Delay Distribution")

    col1, col2 = st.columns(2)

    col1.metric(
        "Not Delayed",
        f"{stats['not_delayed'] / stats['total_flights']:.1%}"
    )

    col2.metric(
        "Delayed",
        f"{stats['delay_rate']:.1%}"
    )

    target_table = pd.DataFrame(
        {
            "Outcome": [
                "Not Delayed",
                "Delayed"
            ],
            "Flights": [
                stats["not_delayed"],
                stats["delayed"]
            ],
            "Percentage": [
                stats["not_delayed"] / stats["total_flights"] * 100,
                stats["delay_rate"] * 100
            ]
        }
    )

    st.dataframe(
        target_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # AIRLINE
    # --------------------------------------------------------

    st.subheader("✈️ Airline Delay Analysis")

    airline_df = load_report_csv(
        "airline_delay_analysis.csv"
    )

    if not airline_df.empty:

        chart_path = REPORTS_PATH / "airline_delay_rate.png"

        col1, col2 = st.columns(2)

        with col1:

            if chart_path.exists():
                st.image(
                    str(chart_path),
                    use_container_width=True
                )

        with col2:

            st.dataframe(
                airline_df,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # MONTH
    # --------------------------------------------------------

    st.subheader("📅 Monthly Delay Analysis")

    monthly_df = load_report_csv(
        "monthly_delay_analysis.csv"
    )

    chart_path = REPORTS_PATH / "monthly_delay_rate.png"

    col1, col2 = st.columns(2)

    with col1:

        if chart_path.exists():

            st.image(
                str(chart_path),
                use_container_width=True
            )

    with col2:

        if not monthly_df.empty:

            st.dataframe(
                monthly_df,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # DAY OF WEEK
    # --------------------------------------------------------

    st.subheader("📆 Day of Week Delay Analysis")

    dow_df = load_report_csv(
        "day_of_week_delay_analysis.csv"
    )

    chart_path = REPORTS_PATH / "day_of_week_delay_rate.png"

    col1, col2 = st.columns(2)

    with col1:

        if chart_path.exists():

            st.image(
                str(chart_path),
                use_container_width=True
            )

    with col2:

        if not dow_df.empty:

            st.dataframe(
                dow_df,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # DEPARTURE HOUR
    # --------------------------------------------------------

    st.subheader("🕐 Departure Hour Analysis")

    hour_df = load_report_csv(
        "departure_hour_delay_analysis.csv"
    )

    chart_path = REPORTS_PATH / "departure_hour_delay_rate.png"

    col1, col2 = st.columns(2)

    with col1:

        if chart_path.exists():

            st.image(
                str(chart_path),
                use_container_width=True
            )

    with col2:

        if not hour_df.empty:

            st.dataframe(
                hour_df,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # TIME OF DAY
    # --------------------------------------------------------

    st.subheader("🌅 Time of Day Analysis")

    tod_df = load_report_csv(
        "time_of_day_delay_analysis.csv"
    )

    chart_path = REPORTS_PATH / "time_of_day_delay_rate.png"

    col1, col2 = st.columns(2)

    with col1:

        if chart_path.exists():

            st.image(
                str(chart_path),
                use_container_width=True
            )

    with col2:

        if not tod_df.empty:

            st.dataframe(
                tod_df,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # ORIGIN AIRPORT
    # --------------------------------------------------------

    st.subheader("🛫 Origin Airport Delay Analysis")

    origin_df = load_report_csv(
        "origin_airport_delay_analysis.csv"
    )

    if not origin_df.empty:

        st.write("Top 15 origin airports by delay rate")

        origin_rate_col = find_column(
            origin_df,
            [
                "DELAY_RATE",
                "DELAY_PERCENTAGE",
                "DELAY_PERCENT",
                "PERCENT_DELAYED"
            ]
        )

        if origin_rate_col:

            top_origin = (
                origin_df
                .sort_values(
                    origin_rate_col,
                    ascending=False
                )
                .head(15)
            )

            st.dataframe(
                top_origin,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.dataframe(
                origin_df.head(15),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # DESTINATION AIRPORT
    # --------------------------------------------------------

    st.subheader("🛬 Destination Airport Delay Analysis")

    destination_df = load_report_csv(
        "destination_airport_delay_analysis.csv"
    )

    if not destination_df.empty:

        st.write(
            "Top 15 destination airports by delay rate"
        )

        destination_rate_col = find_column(
            destination_df,
            [
                "DELAY_RATE",
                "DELAY_PERCENTAGE",
                "DELAY_PERCENT",
                "PERCENT_DELAYED"
            ]
        )

        if destination_rate_col:

            top_destination = (
                destination_df
                .sort_values(
                    destination_rate_col,
                    ascending=False
                )
                .head(15)
            )

            st.dataframe(
                top_destination,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.dataframe(
                destination_df.head(15),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# AIRPORTS & ROUTES
# ============================================================

elif page == "🗺️ Airports & Routes":

    st.title("🗺️ Airports & Route Analysis")

    st.write(
        """
        Explore airport codes, flight volumes, route frequency
        and recurring delay patterns.
        """
    )

    st.markdown("---")

    # ========================================================
    # AIRPORT DIRECTORY
    # ========================================================

    st.subheader("🏢 Airport Code Directory")

    airports = load_airports()

    if not airports.empty:

        st.write(
            "Search, sort and filter the airport directory below."
        )

        st.dataframe(
            airports,
            use_container_width=True,
            hide_index=True,
            height=500
        )

    else:

        st.warning(
            """
            airports.csv could not be found.

            Expected locations include:

            data/airports.csv

            or

            data/raw/airports.csv
            """
        )

    st.markdown("---")

    # ========================================================
    # ROUTE ANALYSIS
    # ========================================================

    st.subheader("🛣️ Route Delay Analysis")

    route_df = load_report_csv(
        "route_delay_analysis.csv"
    )

    if route_df.empty:

        st.warning(
            "route_delay_analysis.csv was not found."
        )

    else:

        # ----------------------------------------------
        # Display complete route analysis
        # ----------------------------------------------

        st.write(
            "Complete route-level delay analysis:"
        )

        st.dataframe(
            route_df,
            use_container_width=True,
            hide_index=True,
            height=500
        )

        # ----------------------------------------------
        # Identify columns
        # ----------------------------------------------

        route_col = find_column(
            route_df,
            [
                "ROUTE",
                "ROUTE_CODE"
            ]
        )

        flights_col = find_column(
            route_df,
            [
                "FLIGHTS",
                "FLIGHT_COUNT",
                "TOTAL_FLIGHTS",
                "VOLUME",
                "COUNT"
            ]
        )

        delayed_col = find_column(
            route_df,
            [
                "DELAYED_FLIGHTS",
                "DELAYED",
                "DELAY_COUNT"
            ]
        )

        rate_col = find_column(
            route_df,
            [
                "DELAY_RATE",
                "DELAY_PERCENTAGE",
                "DELAY_PERCENT",
                "PERCENT_DELAYED"
            ]
        )

        # ====================================================
        # ROUTE FREQUENCY
        # ====================================================

        st.markdown("---")

        st.subheader("📊 Route Frequency / Flight Volume")

        if route_col and flights_col:

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "### Most Frequent Routes"
                )

                top_volume = (
                    route_df
                    .sort_values(
                        flights_col,
                        ascending=False
                    )
                    .head(15)
                )

                st.dataframe(
                    top_volume[
                        [
                            route_col,
                            flights_col
                        ]
                        + (
                            [delayed_col]
                            if delayed_col
                            else []
                        )
                        + (
                            [rate_col]
                            if rate_col
                            else []
                        )
                    ],
                    use_container_width=True,
                    hide_index=True
                )

            with col2:

                st.markdown(
                    "### Least Frequent Routes"
                )

                bottom_volume = (
                    route_df
                    .sort_values(
                        flights_col,
                        ascending=True
                    )
                    .head(15)
                )

                st.dataframe(
                    bottom_volume[
                        [
                            route_col,
                            flights_col
                        ]
                        + (
                            [delayed_col]
                            if delayed_col
                            else []
                        )
                        + (
                            [rate_col]
                            if rate_col
                            else []
                        )
                    ],
                    use_container_width=True,
                    hide_index=True
                )

        # ====================================================
        # ROUTES WITH HIGHEST DELAY RATE
        # ====================================================

        if rate_col:

            st.markdown("---")

            st.subheader(
                "⚠️ Routes with Highest Delay Rates"
            )

            st.write(
                """
                A minimum flight-volume filter is used so that
                routes with only a handful of flights do not
                dominate the analysis.
                """
            )

            if flights_col:

                min_volume = st.slider(
                    "Minimum number of flights",
                    min_value=1,
                    max_value=1000,
                    value=100,
                    step=25
                )

                filtered_routes = route_df[
                    route_df[flights_col] >= min_volume
                ]

            else:

                filtered_routes = route_df

            top_delay_routes = (
                filtered_routes
                .sort_values(
                    rate_col,
                    ascending=False
                )
                .head(20)
            )

            st.dataframe(
                top_delay_routes,
                use_container_width=True,
                hide_index=True
            )

        # ====================================================
        # ROUTE VOLUME + DELAY
        # ====================================================

        if flights_col and rate_col:

            st.markdown("---")

            st.subheader(
                "📈 Route Volume vs Delay Rate"
            )

            st.write(
                """
                This view helps distinguish high-delay routes
                from high-volume routes. A route can have a high
                delay rate but very little traffic, while another
                route may have a moderate delay rate across a
                very large number of flights.
                """
            )

            volume_table = (
                route_df[
                    [
                        route_col,
                        flights_col,
                        rate_col
                    ]
                ]
                .sort_values(
                    flights_col,
                    ascending=False
                )
                .head(30)
            )

            st.dataframe(
                volume_table,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# PREDICTION
# ============================================================

elif page == "✈️ Prediction":

    st.title("✈️ Flight Delay Prediction")

    st.write(
        """
        Enter the flight details below to estimate the probability
        that the flight will be delayed.
        """
    )

    st.markdown("---")

    st.subheader("Flight Information")

    # --------------------------------------------------------
    # Get known values for dropdowns
    # --------------------------------------------------------

    airline_values = []

    origin_values = []

    destination_values = []

    df_for_inputs = load_data()

    if "AIRLINE" in df_for_inputs.columns:

        airline_values = sorted(
            df_for_inputs["AIRLINE"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    if "ORIGIN_AIRPORT" in df_for_inputs.columns:

        origin_values = sorted(
            df_for_inputs["ORIGIN_AIRPORT"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    if "DESTINATION_AIRPORT" in df_for_inputs.columns:

        destination_values = sorted(
            df_for_inputs["DESTINATION_AIRPORT"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=8
        )

        day = st.number_input(
            "Day",
            min_value=1,
            max_value=31,
            value=15
        )

        day_of_week = st.number_input(
            "Day of Week",
            min_value=1,
            max_value=7,
            value=6
        )

        if airline_values:

            airline = st.selectbox(
                "Airline Code",
                airline_values
            )

        else:

            airline = st.text_input(
                "Airline Code",
                value="DL"
            )

    with col2:

        if origin_values:

            origin_airport = st.selectbox(
                "Origin Airport",
                origin_values
            )

        else:

            origin_airport = st.text_input(
                "Origin Airport",
                value="ATL"
            )

        if destination_values:

            destination_airport = st.selectbox(
                "Destination Airport",
                destination_values
            )

        else:

            destination_airport = st.text_input(
                "Destination Airport",
                value="JFK"
            )

        scheduled_departure = st.number_input(
            "Scheduled Departure",
            min_value=0,
            max_value=2359,
            value=1800
        )

        scheduled_time = st.number_input(
            "Scheduled Flight Time (minutes)",
            min_value=1,
            value=150
        )

    with col3:

        distance = st.number_input(
            "Distance (miles)",
            min_value=1,
            value=760
        )

        scheduled_arrival = st.number_input(
            "Scheduled Arrival",
            min_value=0,
            max_value=2359,
            value=2030
        )

        departure_hour = st.number_input(
            "Departure Hour",
            min_value=0,
            max_value=23,
            value=18
        )

        time_of_day = st.selectbox(
            "Time of Day",
            [
                "Morning",
                "Afternoon",
                "Evening",
                "Night"
            ]
        )

    st.markdown("---")

    predict_button = st.button(
        "✈️ Predict Flight Delay",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        flight = pd.DataFrame(
            [
                {
                    "MONTH": month,
                    "DAY": day,
                    "DAY_OF_WEEK": day_of_week,
                    "AIRLINE": airline,
                    "ORIGIN_AIRPORT": origin_airport,
                    "DESTINATION_AIRPORT": destination_airport,
                    "SCHEDULED_DEPARTURE": scheduled_departure,
                    "SCHEDULED_TIME": scheduled_time,
                    "DISTANCE": distance,
                    "SCHEDULED_ARRIVAL": scheduled_arrival,
                    "DEPARTURE_HOUR": departure_hour,
                    "TIME_OF_DAY": time_of_day
                }
            ]
        )

        prediction = model.predict(flight)[0]

        probabilities = model.predict_proba(flight)[0]

        no_delay_probability = probabilities[0]

        delay_probability = probabilities[1]

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                "⚠️ FLIGHT LIKELY TO BE DELAYED"
            )

        else:

            st.success(
                "✅ FLIGHT LIKELY TO BE ON TIME"
            )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Delay Probability",
                f"{delay_probability:.2%}"
            )

        with col2:

            st.metric(
                "No-Delay Probability",
                f"{no_delay_probability:.2%}"
            )

        st.progress(
            float(delay_probability)
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    st.write(
        "Comparison and evaluation of the machine learning models."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("📊 Model Comparison")

    comparison_path = (
        REPORTS_PATH / "model_comparison.csv"
    )

    if comparison_path.exists():

        results = pd.read_csv(
            comparison_path
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Model comparison file not found."
        )

    st.markdown("---")

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    st.subheader("🎯 Confusion Matrices")

    models = [
        (
            "Logistic Regression",
            "confusion_matrix_logistic_regression.png"
        ),
        (
            "Random Forest",
            "confusion_matrix_random_forest.png"
        ),
        (
            "Linear SVM",
            "confusion_matrix_linear_svm.png"
        )
    ]

    cols = st.columns(3)

    for col, (name, filename) in zip(
        cols,
        models
    ):

        with col:

            st.markdown(
                f"**{name}**"
            )

            image_path = (
                REPORTS_PATH / filename
            )

            if image_path.exists():

                st.image(
                    str(image_path),
                    use_container_width=True
                )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL CURVES
    # --------------------------------------------------------
    st.subheader("📈 Model Curves")

    col1, col2 = st.columns(2)

    # ROC Curve
    with col1:

        st.markdown("#### ROC Curve")

        roc_path = (
            REPORTS_PATH / "roc_curve_comparison.png"
        )

        if roc_path.exists():

            st.image(
                str(roc_path),
                use_container_width=True
            )

            with open(roc_path, "rb") as f:

                st.download_button(
                    "⬇️ Download ROC Curve",
                    data=f,
                    file_name="roc_curve_comparison.png",
                    mime="image/png",
                    use_container_width=True,
                    key="download_roc_curve"
                )

    # Precision-Recall Curve
    with col2:

        st.markdown("#### Precision-Recall Curve")

        pr_path = (
            REPORTS_PATH
            / "precision_recall_curve_comparison.png"
        )

        if pr_path.exists():

            st.image(
                str(pr_path),
                use_container_width=True
            )

            with open(pr_path, "rb") as f:

                st.download_button(
                    "⬇️ Download Precision-Recall Curve",
                    data=f,
                    file_name="precision_recall_curve_comparison.png",
                    mime="image/png",
                    use_container_width=True,
                    key="download_pr_curve"
                )

# ============================================================
# REPORT GENERATION FUNCTIONS
# ============================================================

def create_pdf_report():

    if not REPORTLAB_AVAILABLE:

        return None

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    story = []

    story.append(
        Paragraph(
            "Flight Delay Prediction Report",
            title_style
        )
    )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Machine Learning Classification Project",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            f"Total flights: {stats['total_flights']:,}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Airlines: {stats['airlines']}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Airports: {stats['airports']}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Unique routes: {stats['routes']:,}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Overall delay rate: {stats['delay_rate']:.2%}",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Dataset Split",
            styles["Heading2"]
        )
    )

    split_data = [
        ["Dataset", "Flights", "Percentage"],
        ["Training", "4,571,206", "80%"],
        ["Testing", "1,142,802", "20%"]
    ]

    table = Table(split_data)

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER")
            ]
        )
    )

    story.append(table)

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Prediction Target",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            "IS_DELAYED = 1 indicates a flight delayed by more "
            "than 15 minutes. IS_DELAYED = 0 indicates that "
            "the flight was not delayed.",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Machine Learning Models",
            styles["Heading2"]
        )
    )

    model_data = [
        ["Model"],
        ["Logistic Regression"],
        ["Random Forest"],
        ["Linear SVM"]
    ]

    model_table = Table(model_data)

    model_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 1, colors.black)
            ]
        )
    )

    story.append(model_table)

    # Model comparison

    comparison = load_report_csv(
        "model_comparison.csv"
    )

    if not comparison.empty:

        story.append(
            Spacer(1, 20)
        )

        story.append(
            Paragraph(
                "Model Comparison",
                styles["Heading2"]
            )
        )

        pdf_data = [
            list(comparison.columns)
        ]

        for row in comparison.itertuples(
            index=False
        ):

            pdf_data.append(
                [
                    str(value)
                    for value in row
                ]
            )

        pdf_table = Table(
            pdf_data,
            repeatRows=1
        )

        pdf_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.grey
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7
                    )
                ]
            )
        )

        story.append(pdf_table)

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# EXCEL REPORT
# ============================================================

def create_excel_report():

    if not OPENPYXL_AVAILABLE:

        return None

    buffer = BytesIO()

    wb = Workbook()

    ws = wb.active
    ws.title = "Overview"

    ws.append(
        [
            "Statistic",
            "Value"
        ]
    )

    overview_rows = [
        (
            "Total Flights",
            stats["total_flights"]
        ),
        (
            "Airlines",
            stats["airlines"]
        ),
        (
            "Airports",
            stats["airports"]
        ),
        (
            "Unique Routes",
            stats["routes"]
        ),
        (
            "Delayed Flights",
            stats["delayed"]
        ),
        (
            "Non-Delayed Flights",
            stats["not_delayed"]
        ),
        (
            "Delay Rate",
            stats["delay_rate"]
        ),
        (
            "Average Distance",
            stats.get(
                "avg_distance",
                0
            )
        ),
        (
            "Average Scheduled Time",
            stats.get(
                "avg_scheduled_time",
                0
            )
        )
    ]

    for row in overview_rows:

        ws.append(row)

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    comparison = load_report_csv(
        "model_comparison.csv"
    )

    if not comparison.empty:

        ws2 = wb.create_sheet(
            "Model Comparison"
        )

        ws2.append(
            list(comparison.columns)
        )

        for row in comparison.itertuples(
            index=False
        ):

            ws2.append(
                list(row)
            )

    # --------------------------------------------------------
    # ALL ANALYSIS FILES
    # --------------------------------------------------------

    analysis_files = [
        (
            "Airline Analysis",
            "airline_delay_analysis.csv"
        ),
        (
            "Monthly Analysis",
            "monthly_delay_analysis.csv"
        ),
        (
            "Day Analysis",
            "day_of_week_delay_analysis.csv"
        ),
        (
            "Hour Analysis",
            "departure_hour_delay_analysis.csv"
        ),
        (
            "Time Analysis",
            "time_of_day_delay_analysis.csv"
        ),
        (
            "Origin Analysis",
            "origin_airport_delay_analysis.csv"
        ),
        (
            "Destination Analysis",
            "destination_airport_delay_analysis.csv"
        ),
        (
            "Route Analysis",
            "route_delay_analysis.csv"
        )
    ]

    for sheet_name, filename in analysis_files:

        df = load_report_csv(filename)

        if df.empty:
            continue

        ws_analysis = wb.create_sheet(
            sheet_name[:31]
        )

        ws_analysis.append(
            list(df.columns)
        )

        for row in df.itertuples(
            index=False
        ):

            ws_analysis.append(
                list(row)
            )

    # --------------------------------------------------------
    # AIRPORT DIRECTORY
    # --------------------------------------------------------

    airports = load_airports()

    if not airports.empty:

        ws_airports = wb.create_sheet(
            "Airport Directory"
        )

        ws_airports.append(
            list(airports.columns)
        )

        for row in airports.itertuples(
            index=False
        ):

            ws_airports.append(
                list(row)
            )

    wb.save(buffer)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# POWERPOINT REPORT
# ============================================================

def create_ppt_report():

    if not PPTX_AVAILABLE:

        return None

    prs = Presentation()

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    slide = prs.slides.add_slide(
        prs.slide_layouts[0]
    )

    slide.shapes.title.text = (
        "Flight Delay Prediction"
    )

    slide.placeholders[1].text = (
        "Machine Learning Analytics Project"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    slide = prs.slides.add_slide(
        prs.slide_layouts[1]
    )

    slide.shapes.title.text = (
        "Dataset Overview"
    )

    body = slide.placeholders[1].text_frame

    body.text = (
        f"Total flights: {stats['total_flights']:,}"
    )

    for text in [
        f"Airlines: {stats['airlines']}",
        f"Airports: {stats['airports']}",
        f"Unique routes: {stats['routes']:,}",
        f"Delay rate: {stats['delay_rate']:.2%}"
    ]:

        p = body.add_paragraph()
        p.text = text

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    slide = prs.slides.add_slide(
        prs.slide_layouts[1]
    )

    slide.shapes.title.text = (
        "Dataset Split"
    )

    body = slide.placeholders[1].text_frame

    body.text = (
        "80% Training Data"
    )

    p = body.add_paragraph()
    p.text = "20% Testing Data"

    p = body.add_paragraph()
    p.text = (
        "Stratified train-test split"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    slide = prs.slides.add_slide(
        prs.slide_layouts[1]
    )

    slide.shapes.title.text = (
        "Models Used"
    )

    body = slide.placeholders[1].text_frame

    body.text = "Logistic Regression"

    for text in [
        "Random Forest",
        "Linear SVM"
    ]:

        p = body.add_paragraph()
        p.text = text

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    comparison = load_report_csv(
        "model_comparison.csv"
    )

    if not comparison.empty:

        slide = prs.slides.add_slide(
            prs.slide_layouts[5]
        )

        slide.shapes.title.text = (
            "Model Comparison"
        )

        rows = len(comparison) + 1

        cols = len(comparison.columns)

        table = slide.shapes.add_table(
            rows,
            cols,
            Inches(0.5),
            Inches(1.5),
            Inches(9),
            Inches(4)
        ).table

        for j, col in enumerate(
            comparison.columns
        ):

            table.cell(
                0,
                j
            ).text = str(col)

        for i, row in enumerate(
            comparison.itertuples(
                index=False
            ),
            start=1
        ):

            for j, value in enumerate(row):

                table.cell(
                    i,
                    j
                ).text = str(value)

    buffer = BytesIO()

    prs.save(buffer)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# REPORTS & EXPORT
# ============================================================

if page == "📄 Reports & Export":

    st.title("📄 Reports & Export")

    st.write(
        """
        Review the project findings and download the analysis
        in PDF, Excel, PowerPoint or CSV format.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # REPORT REVIEW
    # --------------------------------------------------------

    st.subheader("📋 Report Review")

    st.markdown(
        f"""
        ### Flight Delay Prediction Project

        **Dataset:** {stats['total_flights']:,} flights

        **Airlines:** {stats['airlines']}

        **Airports:** {stats['airports']}

        **Unique Routes:** {stats['routes']:,}

        **Overall Delay Rate:** {stats['delay_rate']:.2%}

        ### Dataset Split

        - Training: **4,571,206 flights (80%)**
        - Testing: **1,142,802 flights (20%)**
        - Stratified split

        ### Prediction Target

        `IS_DELAYED = 1` means the flight was delayed by more
        than 15 minutes.

        ### Models

        - Logistic Regression
        - Random Forest
        - Linear SVM

        ### Analysis Covered

        - Airline delay patterns
        - Monthly delay patterns
        - Day-of-week patterns
        - Departure-hour patterns
        - Time-of-day patterns
        - Origin airport analysis
        - Destination airport analysis
        - Route delay analysis
        - Route flight volume
        - Model comparison
        - Confusion matrices
        - ROC curve
        - Precision-Recall curve
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # HTML REPORT
    # --------------------------------------------------------

    st.subheader("📄 Project Report")

    report_title = "Flight Delay Prediction - ML Project Report"

    html_report = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>{report_title}</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    color: #172033;
    background: #f5f7fa;
    line-height: 1.6;
}}

.container {{
    max-width: 1000px;
    margin: auto;
    background: white;
    padding: 40px;
    border-radius: 12px;
}}

h1 {{
    color: #172033;
    font-size: 32px;
}}

h2 {{
    color: #172033;
    border-bottom: 2px solid #e4e7ec;
    padding-bottom: 8px;
    margin-top: 35px;
}}

.card {{
    background: #f8f9fb;
    border: 1px solid #e4e7ec;
    border-radius: 10px;
    padding: 18px;
    margin: 15px 0;
}}

.metric {{
    font-size: 26px;
    font-weight: bold;
    color: #172033;
}}

.report-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
}}

.report-table th {{
    background: #172033;
    color: white;
    padding: 10px;
    text-align: left;
}}

.report-table td {{
    padding: 9px;
    border-bottom: 1px solid #e4e7ec;
}}

.note {{
    background: #fff8e6;
    border-left: 4px solid #f59e0b;
    padding: 12px;
    margin: 15px 0;
}}

.footer {{
    margin-top: 40px;
    padding-top: 15px;
    border-top: 1px solid #e4e7ec;
    color: #667085;
    font-size: 13px;
}}

</style>

</head>

<body>

<div class="container">

<h1>{report_title}</h1>

<p>
<strong>End-to-End Machine Learning Pipeline Project</strong>
</p>

<h2>1. Project Overview</h2>

<p>
This project develops an end-to-end machine learning pipeline
for predicting whether a scheduled flight will be delayed by
more than 15 minutes.
</p>

<p>
The project covers data ingestion, profiling, validation,
preprocessing, exploratory data analysis, feature engineering,
model training, model comparison, evaluation and prediction.
</p>


<h2>2. Dataset Overview</h2>

<div class="card">

<p>
<strong>Total Flights:</strong>
{stats['total_flights']:,}
</p>

<p>
<strong>Features Used:</strong>
12
</p>

<p>
<strong>Airlines:</strong>
{stats['airlines']}
</p>

<p>
<strong>Airports:</strong>
{stats['airports']}
</p>

<p>
<strong>Unique Routes:</strong>
{stats['routes']:,}
</p>

<p>
<strong>Delayed Flights:</strong>
{stats['delayed']:,}
</p>

<p>
<strong>Overall Delay Rate:</strong>
{stats['delay_rate']:.2%}
</p>

<p>
<strong>Prediction Target:</strong>
IS_DELAYED
</p>

</div>


<h2>3. Prediction Target</h2>

<div class="card">

<p>
<strong>0 → Not delayed</strong>
</p>

<p>
<strong>1 → Delayed by more than 15 minutes</strong>
</p>

</div>


<h2>4. Machine Learning Pipeline</h2>

<ul>

<li>Data Ingestion</li>
<li>Data Profiling</li>
<li>Data Understanding</li>
<li>Schema Definition</li>
<li>Data Validation</li>
<li>Data Preprocessing</li>
<li>Train-Test Split</li>
<li>Exploratory Data Analysis</li>
<li>Feature Engineering</li>
<li>Model Training</li>
<li>Model Evaluation</li>
<li>Model Selection</li>
<li>Prediction</li>
<li>Streamlit Web App</li>

</ul>


<h2>5. Dataset Split</h2>

<table class="report-table">

<tr>
<th>Dataset</th>
<th>Flights</th>
<th>Percentage</th>
</tr>

<tr>
<td>Training Set</td>
<td>4,571,206</td>
<td>80%</td>
</tr>

<tr>
<td>Test Set</td>
<td>1,142,802</td>
<td>20%</td>
</tr>

<tr>
<td>Total</td>
<td>5,714,008</td>
<td>100%</td>
</tr>

</table>

<p>
A stratified 80/20 train-test split was used so that the
proportion of delayed and non-delayed flights was preserved
across both datasets.
</p>


<h2>6. Models Compared</h2>

<ul>

<li>Logistic Regression</li>
<li>Random Forest</li>
<li>Linear SVM</li>

</ul>


<h2>7. Model Performance</h2>

<table class="report-table">

<tr>
<th>Model</th>
<th>Accuracy</th>
<th>Precision</th>
<th>Recall</th>
<th>F1</th>
<th>ROC-AUC</th>
<th>PR-AUC</th>
</tr>

<tr>
<td>Logistic Regression</td>
<td>60.12%</td>
<td>25.49%</td>
<td>65.14%</td>
<td>36.64%</td>
<td>0.6639</td>
<td>0.2811</td>
</tr>

<tr>
<td>Random Forest</td>
<td>55.52%</td>
<td>24.20%</td>
<td>70.96%</td>
<td>36.10%</td>
<td>0.6692</td>
<td>0.2957</td>
</tr>

<tr>
<td>Linear SVM</td>
<td>59.89%</td>
<td>25.42%</td>
<td>65.46%</td>
<td>36.62%</td>
<td>0.6639</td>
<td>0.2812</td>
</tr>

</table>


<h2>8. Exploratory Data Analysis</h2>

<p>
The analysis examined flight delays across airline, month,
day of week, departure hour, time of day, origin airport,
destination airport and route.
</p>

<div class="card">

<p>
<strong>Highest observed airline delay rate:</strong>
NK - 26.50%
</p>

<p>
<strong>Lowest observed airline delay rate:</strong>
HA - 6.87%
</p>

<p>
<strong>Highest monthly delay rate:</strong>
June - 22.75%
</p>

<p>
<strong>Lowest monthly delay rate:</strong>
October - 11.90%
</p>

<p>
<strong>Highest observed route delay rate:</strong>
LGA → MCI - 35.71%
</p>

</div>


<h2>9. Example Prediction</h2>

<div class="card">

<p>
<strong>Prediction:</strong>
DELAYED
</p>

<p>
<strong>Delay Probability:</strong>
60.35%
</p>

<p>
<strong>No-Delay Probability:</strong>
39.65%
</p>

</div>


<h2>10. Interpretation Notes</h2>

<div class="note">

<p>
The prediction represents a probability generated by the trained
machine learning model based on patterns learned from historical
flight data.
</p>

<p>
A predicted delay does not guarantee that an individual flight
will actually be delayed.
</p>

<p>
Because delayed flights represent a smaller proportion of the
dataset, multiple evaluation metrics should be considered rather
than accuracy alone.
</p>

</div>


<h2>11. Conclusion</h2>

<p>
The project demonstrates a complete machine learning workflow
for flight delay prediction, from data preparation and
exploratory analysis through model training, evaluation and
deployment.
</p>

<p>
The Streamlit application provides an interactive interface
for examining flight-delay patterns, comparing model performance
and generating predictions.
</p>


<div class="footer">

Flight Delay Prediction | Machine Learning Pipeline Project

</div>

</div>

</body>

</html>
"""


    # --------------------------------------------------------
    # REPORT PREVIEW
    # --------------------------------------------------------

    st.markdown("### 👀 Report Preview")

    st.components.v1.html(
        html_report,
        height=700,
        scrolling=True
    )


    # --------------------------------------------------------
    # DOWNLOAD HTML REPORT
    # --------------------------------------------------------

    st.download_button(
        label="⬇️ Download HTML Report",
        data=html_report,
        file_name="flight_delay_prediction_report.html",
        mime="text/html",
        use_container_width=True
    )

    # --------------------------------------------------------
    # DOWNLOAD REPORTS
    # --------------------------------------------------------

    st.subheader("📥 Download Reports")

    col1, col2, col3 = st.columns(3)

    # PDF

    with col1:

        pdf_data = create_pdf_report()

        if pdf_data:

            st.download_button(
                "📄 Download PDF Report",
                data=pdf_data,
                file_name=(
                    "flight_delay_report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )

       

    # Excel

    with col2:

        excel_data = create_excel_report()

        if excel_data:

            st.download_button(
                "📊 Download Excel Report",
                data=excel_data,
                file_name=(
                    "flight_delay_analysis.xlsx"
                ),
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True
            )

       

    # PowerPoint

    with col3:

        ppt_data = create_ppt_report()

        if ppt_data:

            st.download_button(
                "📽️ Download PowerPoint",
                data=ppt_data,
                file_name=(
                    "flight_delay_report.pptx"
                ),
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "presentationml.presentation"
                ),
                use_container_width=True
            )

    st.markdown("---")

    # --------------------------------------------------------
    # CSV FILES
    # --------------------------------------------------------

    st.subheader(
        "📑 Individual CSV Downloads"
    )

    csv_files = [
        (
            "Model Comparison",
            "model_comparison.csv"
        ),
        (
            "Airline Analysis",
            "airline_delay_analysis.csv"
        ),
        (
            "Monthly Analysis",
            "monthly_delay_analysis.csv"
        ),
        (
            "Day of Week Analysis",
            "day_of_week_delay_analysis.csv"
        ),
        (
            "Departure Hour Analysis",
            "departure_hour_delay_analysis.csv"
        ),
        (
            "Time of Day Analysis",
            "time_of_day_delay_analysis.csv"
        ),
        (
            "Origin Airport Analysis",
            "origin_airport_delay_analysis.csv"
        ),
        (
            "Destination Airport Analysis",
            "destination_airport_delay_analysis.csv"
        ),
        (
            "Route Analysis",
            "route_delay_analysis.csv"
        )
    ]

    for display_name, filename in csv_files:

        path = REPORTS_PATH / filename

        if path.exists():

            with open(
                path,
                "rb"
            ) as file:

                st.download_button(
                    f"📥 {display_name}",
                    data=file,
                    file_name=filename,
                    mime="text/csv"
                )


# ============================================================
# GLOSSARY
# ============================================================

elif page == "📖 Glossary":

    st.title("📖 ML Glossary")

    glossary = {

        "Accuracy":
        "Percentage of all predictions that were correct.",

        "Precision":
        "Of the flights predicted as delayed, the proportion "
        "that were actually delayed.",

        "Recall":
        "Of the flights that were actually delayed, the "
        "proportion correctly identified by the model.",

        "F1 Score":
        "Harmonic mean of precision and recall.",

        "ROC-AUC":
        "Measures how well the model separates delayed and "
        "non-delayed flights across classification thresholds.",

        "PR-AUC":
        "Summarizes the precision-recall relationship and is "
        "particularly useful when classes are imbalanced.",

        "Confusion Matrix":
        "Table showing true positives, true negatives, "
        "false positives and false negatives.",

        "Feature":
        "An input variable used by the machine learning model.",

        "Target":
        "The outcome that the model is trying to predict.",

        "Classification":
        "A machine learning task where observations are "
        "assigned to categories.",

        "One-Hot Encoding":
        "A method for converting categorical variables into "
        "numerical columns.",

        "Random Forest":
        "An ensemble machine learning algorithm that combines "
        "multiple decision trees.",

        "Logistic Regression":
        "A classification algorithm that estimates the "
        "probability of an outcome.",

        "SVM":
        "Support Vector Machine, a classification algorithm "
        "that finds a decision boundary between classes.",

        "Feature Engineering":
        "Creating or transforming variables so that they are "
        "more useful for machine learning.",

        "Class Imbalance":
        "A situation where one target class occurs much more "
        "frequently than another.",

        "Train-Test Split":
        "Dividing data into a training portion used to build "
        "the model and a testing portion used to evaluate it.",

        "Data Leakage":
        "When information that would not be available at "
        "prediction time accidentally enters the model."
    }

    for term, definition in glossary.items():

        with st.expander(term):

            st.write(definition)


# ============================================================
# SETTINGS
# ============================================================

elif page == "⚙️ Settings":

    st.title("⚙️ Settings")

    st.subheader("🎨 Appearance")

    dark_mode = st.toggle(
        "Dark Mode",
        value=st.session_state.dark_mode
    )

    if dark_mode != st.session_state.dark_mode:

        st.session_state.dark_mode = dark_mode

        st.rerun()

    st.markdown("---")

    st.subheader("ℹ️ Application")

    st.write(
        """
        **Flight Delay Prediction Dashboard**

        Machine Learning Classification Project

        Models:

        - Logistic Regression
        - Random Forest
        - Linear SVM
        """
    )

    st.markdown("---")

    st.subheader("📁 Project Information")

    st.write(
        f"""
        
        **Model:** Random Forest

        **Dataset split:** 80% training / 20% testing
        """
    )