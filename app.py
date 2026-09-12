import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Medicine Alternative Finder", page_icon="💊", layout="centered"
)

# Hardcoded paths to your local artifacts
MODEL_PATH = r"C:\Users\User\Desktop\folders\python\projects\medicine_dataset\medicine_tfidf_model.joblib"
DB_PATH = r"C:\Users\User\Desktop\folders\python\projects\medicine_dataset\processed_medicines_db.joblib"


# Load model & database once (cached for performance)
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load(MODEL_PATH)
    db = joblib.load(DB_PATH)
    return tfidf, db


try:
    tfidf_model, df_clean = load_artifacts()
except Exception as e:
    st.error(
        f"Error loading model artifacts. Make sure paths are correct. Details: {e}"
    )
    st.stop()


# Reusable Recommendation Logic
def get_top_5_alternatives(
    target_medicine_name, max_distance_km=15.0, w_distance=0.3, w_savings=0.4
):
    target_lookup = df_clean[
        df_clean["name"].str.lower() == target_medicine_name.lower()
    ]

    if target_lookup.empty:
        target_lookup = df_clean[
            df_clean["name"]
            .str.lower()
            .str.contains(target_medicine_name.lower(), regex=False)
        ]

    if target_lookup.empty:
        return None, f"Medicine '{target_medicine_name}' was not found."

    target_med = target_lookup.iloc[0]

    # Hard Safety Filter
    candidates = df_clean[
        (df_clean["short_composition1"] == target_med["short_composition1"])
        & (df_clean["short_composition2"] == target_med["short_composition2"])
        & (df_clean["form"] == target_med["form"])
        & (df_clean["Is_discontinued"] == False)
        & (df_clean["name"].str.lower() != target_med["name"].lower())
    ].copy()

    if candidates.empty:
        return (
            None,
            "No active substitutes found with identical active ingredients.",
        )

    # Cost & Distance Filter
    candidates = candidates[
        (candidates["distance_km"] <= max_distance_km)
        & (candidates["price_per_unit"] < target_med["price_per_unit"])
    ].copy()

    if candidates.empty:
        return (
            None,
            f"No cheaper active substitutes found within {max_distance_km} km.",
        )

    # Calculate Savings & Scoring
    candidates["savings_%"] = (
        (target_med["price_per_unit"] - candidates["price_per_unit"])
        / target_med["price_per_unit"]
    ) * 100

    candidate_vectors = tfidf_model.transform(candidates["text_corpus"])
    target_vector = tfidf_model.transform([target_med["text_corpus"]])

    candidates["similarity_score"] = cosine_similarity(
        target_vector, candidate_vectors
    ).flatten()

    norm_dist = 1.0 - (candidates["distance_km"] / max_distance_km)
    norm_savings = candidates["savings_%"] / 100.0

    candidates["composite_score"] = (
        candidates["similarity_score"]
        + (w_distance * norm_dist)
        + (w_savings * norm_savings)
    )

    top_5 = candidates.sort_values(by="composite_score", ascending=False).head(
        5
    )
    return target_med, top_5


# STREAMLIT UI SETUP
st.title("💊 Medicine Alternative Recommender")
st.write(
    "Find cheaper, medically equivalent substitutes with identical active ingredients near you."
)

# Sidebar Controls
st.sidebar.header("Filter Settings")
max_dist = st.sidebar.slider(
    "Maximum Pharmacy Distance (km)",
    min_value=1.0,
    max_value=50.0,
    value=15.0,
    step=1.0,
)

# Medicine Search Bar (Selectbox with search/autocomplete)
all_medicine_names = df_clean["name"].drop_duplicates().tolist()
selected_medicine = st.selectbox(
    "🔍 Search and select a medicine:",
    options=[""] + all_medicine_names,
    index=0,
    placeholder="Type medicine name (e.g., Dolo 650, Augmentin)...",
)

if selected_medicine:
    target_info, results = get_top_5_alternatives(
        selected_medicine, max_distance_km=max_dist
    )

    if target_info is None:
        st.warning(results)
    else:
        # Display Target Info Card
        st.markdown("---")
        st.subheader("🎯 Target Medicine Details")
        col1, col2, col3 = st.columns(3)
        col1.metric("Name", target_info["name"])
        col2.metric("Retail Price", f"₹{target_info['price(₹)']:.2f}")
        col3.metric(
            "Unit Price", f"₹{target_info['price_per_unit']:.2f} / unit"
        )

        st.caption(
            f"**Composition:** {target_info['short_composition1']} | {target_info['short_composition2']}"
        )

        # Display Recommendations
        st.markdown("---")
        st.subheader("💡 Top 5 Cheaper Alternatives")

        for idx, row in results.iterrows():
            with st.container():
                c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                c1.markdown(f"**{row['name']}**\n\n_{row['manufacturer_name']}_")
                c2.markdown(
                    f"**Price:** ₹{row['price(₹)']:.2f}\n\n({row['pack_size_label']})"
                )
                c3.markdown(f"**Distance:** {row['distance_km']:.1f} km")
                c4.markdown(f"🟢 **Savings: {row['savings_%']:.1f}%**")
                st.divider()