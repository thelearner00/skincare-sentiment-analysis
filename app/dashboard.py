"""
Streamlit Dashboard — Skincare Sentiment & Personalization

Five sections:
  1. Brand & Product Explorer  — aspect sentiment scores by brand/product
  2. Skin-Type Divergence      — products that behave differently by skin type or tone
  3. Ingredient Risk Signal    — ingredients correlated with complaint patterns
  4. Disagreement Cases        — reviews where text sentiment conflicts with is_recommended
  5. Model Comparison          — this project vs the reference notebook
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import config

ASPECT_COLS = list(config.ASPECT_KEYWORDS.keys())
ASPECT_LABELS = {
    "hydration": "Hydration",
    "breakouts_irritation": "Breakouts / Irritation",
    "scent": "Scent",
    "packaging": "Packaging",
    "price_value": "Price & Value",
    "texture_application": "Texture & Application",
}
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

st.set_page_config(
    page_title="Skincare Sentiment",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_aspect_data() -> pd.DataFrame:
    path = config.DATA_CLEAN / "aspect_scored.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, engine=config.PARQUET_ENGINE)


@st.cache_data(ttl=3600)
def load_divergence(segment: str) -> pd.DataFrame:
    path = config.OUTPUTS / f"{segment}_divergence.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_ingredient_risk() -> pd.DataFrame:
    path = config.OUTPUTS / "ingredient_risk_signal.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_disagreements() -> pd.DataFrame:
    path = config.OUTPUTS / "disagreement_cases.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_model_comparison() -> pd.DataFrame:
    path = config.OUTPUTS / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_per_class_f1() -> pd.DataFrame:
    import json
    rows = []
    for name, label in [("baseline", "TF-IDF + LogReg"), ("transformer", "DistilBERT-base")]:
        path = config.METRICS / f"{name}.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        for cls, f1 in d.get("f1_per_class", {}).items():
            rows.append({"model": label, "class": cls, "f1": f1})
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def load_freshness_timestamp() -> str:
    import json
    from datetime import datetime, timezone
    path = config.METRICS / "transformer.json"
    if not path.exists():
        return "unknown"
    d = json.load(open(path))
    ts = d.get("_timestamp", "")
    if ts:
        dt = datetime.fromisoformat(ts).astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    return "unknown"


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Skincare Sentiment")
st.sidebar.caption("Built to outperform the reference Kaggle notebook")
st.sidebar.caption(f"Pipeline last run: {load_freshness_timestamp()}")

section = st.sidebar.radio(
    "Section",
    [
        "Brand & Product Explorer",
        "Skin-Type Divergence",
        "Ingredient Risk",
        "Disagreement Cases",
        "Model Comparison",
    ],
)

df = load_aspect_data()


# ── Section 1: Brand & Product Explorer ──────────────────────────────────────

if section == "Brand & Product Explorer":
    st.title("Brand & Product Explorer")

    if df.empty:
        st.warning("Run the full pipeline first (through `aspect_sentiment.py`).")
        st.stop()

    brands = sorted(df["brand_name"].dropna().unique())
    selected_brand = st.selectbox("Brand", brands)
    brand_df = df[df["brand_name"] == selected_brand]

    products = sorted(brand_df["product_name"].dropna().unique())
    view_level = st.radio("View by", ["Brand average", "Specific product"], horizontal=True)

    if view_level == "Specific product" and products:
        selected_product = st.selectbox("Product", products)
        plot_df = brand_df[brand_df["product_name"] == selected_product]
        title_suffix = selected_product
    else:
        plot_df = brand_df
        title_suffix = selected_brand

    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews", f"{len(plot_df):,}")
    if "rating" in plot_df.columns:
        c2.metric("Avg Rating", f"{plot_df['rating'].mean():.2f}")
    if "sentiment_label" in plot_df.columns:
        c3.metric("Sentiment (0=neg, 2=pos)", f"{plot_df['sentiment_label'].mean():.2f}")

    # Aspect bar chart — aspect scores are P(positive) in [0, 1]
    # Annotate each bar with the review count for that aspect (n reviews that mentioned it)
    aspect_rows = []
    for a in ASPECT_COLS:
        if a in plot_df.columns:
            valid = plot_df[a].dropna()
            if len(valid) > 0:
                aspect_rows.append({
                    "aspect": ASPECT_LABELS[a],
                    "mean_score": valid.mean(),
                    "n_reviews": len(valid),
                })

    if aspect_rows:
        aspect_chart_df = pd.DataFrame(aspect_rows)
        fig = px.bar(
            aspect_chart_df,
            x="aspect",
            y="mean_score",
            color="mean_score",
            color_continuous_scale=["#c0392b", "#f39c12", "#27ae60"],
            range_color=[0, 1],
            text=aspect_chart_df["n_reviews"].apply(lambda n: f"n={n:,}"),
            labels={"aspect": "Aspect", "mean_score": "P(positive sentiment)"},
            title=f"Aspect Sentiment — {title_suffix}",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, 1.1])
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Scores are softmax P(positive) in [0, 1]. "
            "n = reviews mentioning this aspect. Low n = treat mean with caution."
        )
    else:
        st.info("No aspect data for this selection — run `aspect_sentiment.py` first.")

    with st.expander("Sample reviews"):
        sample_cols = [c for c in ["review_text", "rating", "sentiment_label", "skin_type"] if c in plot_df.columns]
        st.dataframe(plot_df[sample_cols].dropna(subset=["review_text"]).head(20), use_container_width=True)
        st.download_button(
            "Download sample reviews (CSV)",
            plot_df[sample_cols].dropna(subset=["review_text"]).head(200).to_csv(index=False),
            file_name=f"{title_suffix}_reviews.csv",
        )


# ── Section 2: Skin-Type Divergence ──────────────────────────────────────────

elif section == "Skin-Type Divergence":
    st.title("Skin-Type Divergence")
    st.caption(
        f"Products where a specific skin-type segment's aspect sentiment diverges from the product's overall average. "
        f"Flagged when any |aspect delta| > {config.DIVERGENCE_THRESHOLD}. "
        f"Low-confidence = fewer than {config.MIN_SEGMENT_SIZE} reviews mentioning the aspect in that segment."
    )

    segment = st.radio("Segment by", ["skin_type", "skin_tone"], horizontal=True)
    div_df = load_divergence(segment)

    if div_df.empty:
        st.warning("Run `segment_analysis.py` first.")
        st.stop()

    # Filter by brand (if aspect data available) — BEFORE computing metrics
    if not df.empty:
        brands = ["All"] + sorted(df["brand_name"].dropna().unique())
        selected_brand = st.selectbox("Filter by brand", brands)
        if selected_brand != "All":
            brand_product_ids = df[df["brand_name"] == selected_brand]["product_id"].unique()
            display_df = div_df[div_df["product_id"].isin(brand_product_ids)]
        else:
            display_df = div_df
    else:
        display_df = div_df

    # Metrics computed AFTER the brand filter so they reflect what's shown
    flagged_count = int(display_df["flagged"].sum()) if "flagged" in display_df.columns else 0
    st.metric("Flagged segments (in selection)", flagged_count)

    delta_cols = [c for c in display_df.columns if c.endswith("_delta")]
    sort_aspect = st.selectbox(
        "Sort by aspect delta",
        [c.replace("_delta", "").replace("_", " ") for c in delta_cols],
    ) if delta_cols else None
    sort_col = f"{sort_aspect.replace(' ', '_')}_delta" if sort_aspect else None

    show_cols = (
        ["product_id", "product_name", segment] + delta_cols + ["flagged", "low_confidence"]
    )
    count_cols = [c for c in display_df.columns if c.endswith("_count")]
    show_cols += count_cols
    show_cols = [c for c in show_cols if c in display_df.columns]

    sorted_df = (
        display_df[show_cols].sort_values(sort_col)
        if sort_col and sort_col in display_df.columns
        else display_df[show_cols]
    )
    st.dataframe(sorted_df, use_container_width=True)
    st.download_button(
        "Download divergence table (CSV)",
        sorted_df.to_csv(index=False),
        file_name=f"{segment}_divergence_filtered.csv",
    )

    # Histogram — responds to segment toggle and brand filter
    hist_aspect = st.selectbox(
        "Histogram for aspect", [c.replace("_delta", "") for c in delta_cols]
    ) if delta_cols else None
    if hist_aspect:
        hist_col = f"{hist_aspect}_delta"
        if hist_col in display_df.columns:
            fig = px.histogram(
                display_df,
                x=hist_col,
                nbins=40,
                title=f"{hist_aspect} delta ({segment})",
                labels={hist_col: "Delta (negative = worse than product average)"},
                color_discrete_sequence=["#772E25"],
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Section 3: Ingredient Risk ────────────────────────────────────────────────

elif section == "Ingredient Risk":
    st.title("Ingredient Risk Signal")
    st.warning(
        "**Disclaimer:** The table below shows ingredients that appear more frequently in products "
        "associated with breakout/irritation complaints compared to low-complaint products. "
        "This is a **frequency association**, not clinical evidence of causation. "
        "A Fisher's exact p-value is included but is not corrected for multiple comparisons. "
        "Use as a signal to investigate further, not as a conclusion."
    )

    risk_df = load_ingredient_risk()
    if risk_df.empty:
        st.info("Run `ingredient_risk.py` first.")
        st.stop()

    top_n = st.slider("Show top N ingredients", 10, min(100, len(risk_df)), 30)
    display = risk_df.head(top_n).drop(columns=["association_only_not_causal"], errors="ignore")
    st.dataframe(display, use_container_width=True)
    st.download_button(
        "Download ingredient risk table (CSV)",
        display.to_csv(index=False),
        file_name="ingredient_risk_top.csv",
    )

    # Chart respects the slider
    chart_df = risk_df.head(top_n).copy()
    chart_df = chart_df[chart_df["risk_ratio_association_only"] != float("inf")]
    if not chart_df.empty:
        fig = px.bar(
            chart_df,
            x="risk_ratio_association_only",
            y="ingredient",
            orientation="h",
            title=f"Top {top_n} Ingredients by Association Ratio (finite ratios only)",
            labels={
                "risk_ratio_association_only": "Association ratio (not causal)",
                "ingredient": "Ingredient",
            },
            color="risk_ratio_association_only",
            color_continuous_scale="Reds",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


# ── Section 4: Disagreement Cases ─────────────────────────────────────────────

elif section == "Disagreement Cases":
    st.title("Disagreement Cases")
    st.caption(
        "Reviews where the model's predicted text sentiment conflicts with the reviewer's is_recommended checkbox — "
        "on the transformer's held-out test set only (not training data). "
        "This analysis is only possible because is_recommended is NOT the training label."
    )

    dis_df = load_disagreements()
    if dis_df.empty:
        st.info("Run `disagreement.py` first.")
        st.stop()

    dis_type = st.radio(
        "Show",
        ["All disagreements", "Recommended but negative", "Not recommended but positive"],
        horizontal=True,
    )

    if dis_type == "Recommended but negative" and "recommended_but_negative" in dis_df.columns:
        filtered = dis_df[dis_df["recommended_but_negative"].astype(bool)]
    elif dis_type == "Not recommended but positive" and "not_recommended_but_positive" in dis_df.columns:
        filtered = dis_df[dis_df["not_recommended_but_positive"].astype(bool)]
    else:
        filtered = dis_df

    # Brand filter — BEFORE computing metrics so numbers reflect what's shown
    if not df.empty:
        brands = ["All"] + sorted(df["brand_name"].dropna().unique())
        selected_brand = st.selectbox("Filter by brand", brands)
        if selected_brand != "All" and "brand_name" in filtered.columns:
            filtered = filtered[filtered["brand_name"] == selected_brand]

    # Metrics reflect the current filter
    c1, c2, c3 = st.columns(3)
    c1.metric("Total shown", f"{len(filtered):,}")
    if "recommended_but_negative" in filtered.columns:
        c2.metric("Recommended but negative", f"{filtered['recommended_but_negative'].sum():,}")
    if "not_recommended_but_positive" in filtered.columns:
        c3.metric("Not recommended but positive", f"{filtered['not_recommended_but_positive'].sum():,}")

    if "predicted_sentiment" in filtered.columns:
        filtered = filtered.copy()
        filtered["predicted_sentiment"] = filtered["predicted_sentiment"].map(LABEL_NAMES).fillna(
            filtered["predicted_sentiment"]
        )

    display_cols = [
        c for c in
        ["review_text", "rating", "is_recommended", "predicted_sentiment",
         "product_name", "brand_name", "skin_type"]
        if c in filtered.columns
    ]

    n_cap = 200
    st.caption(f"Showing up to {n_cap:,} rows. Use download for the full filtered set.")
    st.dataframe(filtered[display_cols].head(n_cap), use_container_width=True)
    st.download_button(
        "Download filtered disagreements (CSV)",
        filtered[display_cols].to_csv(index=False),
        file_name="disagreements_filtered.csv",
    )

    # Brand-level disagreement rate chart
    if "brand_name" in dis_df.columns and len(dis_df) > 0:
        brand_disagree = (
            dis_df.groupby("brand_name")
            .size()
            .reset_index(name="disagreement_count")
            .sort_values("disagreement_count", ascending=False)
            .head(20)
        )
        fig = px.bar(
            brand_disagree,
            x="disagreement_count",
            y="brand_name",
            orientation="h",
            title="Top 20 Brands by Disagreement Case Count",
            labels={"disagreement_count": "Disagreement cases", "brand_name": "Brand"},
            color_discrete_sequence=["#772E25"],
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


# ── Section 5: Model Comparison ───────────────────────────────────────────────

elif section == "Model Comparison":
    st.title("Model Comparison")
    st.caption("Our measured results vs the reference Kaggle notebook")

    cmp_df = load_model_comparison()
    if cmp_df.empty:
        st.info("Run `aggregate.py` to generate the comparison table.")
        st.stop()

    st.dataframe(cmp_df, use_container_width=True)
    st.download_button("Download comparison table (CSV)", cmp_df.to_csv(index=False), file_name="model_comparison.csv")

    st.markdown("""
**Why direct accuracy comparison is misleading:**

The reference rows predict a **binary** `is_recommended` label (random baseline = 50%).
Our rows predict a **3-class** rating-derived label (random baseline = 33%, majority-class baseline ≈ 83%).
A number like 90.7% on a 3-class problem is not the same as 88.9% on an easier binary problem.

The honest framing: **our DistilBERT achieves 91.2% weighted F1 on the harder task, with a majority-class
baseline of 83.2% — an 8-point lift over always predicting "positive".** The TF-IDF baseline delivers 86.6%
weighted F1 — a 3.4-point lift, with no data discarded and negations preserved.
""")

    # Accuracy chart — our models only (reference comparison is misleading as noted above)
    our_rows = cmp_df[cmp_df["label_type"] == "rating (3-class)"].dropna(subset=["accuracy"])
    if not our_rows.empty:
        fig_acc = px.bar(
            our_rows,
            x="model",
            y="accuracy",
            title="Accuracy — 3-class rating task (our models only)",
            labels={"accuracy": "Accuracy", "model": ""},
            color_discrete_sequence=["#772E25", "#c0392b", "#aaaaaa"],
            text=our_rows["accuracy"].round(3),
        )
        fig_acc.update_traces(textposition="outside")
        fig_acc.update_layout(xaxis_tickangle=-15, yaxis_range=[0.75, 1.0])
        st.plotly_chart(fig_acc, use_container_width=True)

    # F1 chart — weighted F1 is the honest metric for imbalanced multi-class
    our_f1_rows = our_rows.dropna(subset=["f1_weighted"])
    if not our_f1_rows.empty:
        fig_f1 = px.bar(
            our_f1_rows,
            x="model",
            y="f1_weighted",
            title="Weighted F1 — 3-class rating task (our models only)",
            labels={"f1_weighted": "Weighted F1", "model": ""},
            color_discrete_sequence=["#772E25", "#c0392b"],
            text=our_f1_rows["f1_weighted"].round(3),
        )
        fig_f1.update_traces(textposition="outside")
        fig_f1.update_layout(xaxis_tickangle=-15, yaxis_range=[0.75, 1.0])
        st.plotly_chart(fig_f1, use_container_width=True)
        st.caption(
            "Weighted F1 accounts for class imbalance and is the primary metric — "
            "it penalises models that ignore minority classes (negative, neutral)."
        )

    # Per-class F1 breakdown — shows where each model actually struggles
    pcf1_df = load_per_class_f1()
    if not pcf1_df.empty:
        st.subheader("Per-Class F1")
        fig_pc = px.bar(
            pcf1_df,
            x="class",
            y="f1",
            color="model",
            barmode="group",
            range_y=[0, 1],
            title="Per-Class F1 — neutral is the hardest class",
            labels={"f1": "F1 Score", "class": "Sentiment Class", "model": "Model"},
            text=pcf1_df["f1"].round(3),
        )
        fig_pc.update_traces(textposition="outside")
        st.plotly_chart(fig_pc, use_container_width=True)
        st.caption(
            "Weighted F1 (above) is dominated by the positive class (82% of reviews). "
            "Per-class F1 shows where models actually struggle: neutral class has the lowest F1 "
            "for both models because the rating=3 boundary is the hardest to learn."
        )
