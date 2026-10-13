
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from predict import load_model, predict_sentiment
from complaint_themes import identify_themes
from src.recommendations import get_recommendation


st.set_page_config(
    page_title="CampusPulse AI",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 CampusPulse AI")
st.subheader("AI-powered Student Feedback Intelligence")
st.write(
    "Analyze student feedback, understand sentiment, "
    "and identify recurring areas for improvement."
)

st.sidebar.title("CampusPulse AI")
st.sidebar.caption("Student Feedback Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Analyze Feedback", "About the Project"],
)


@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.info("Run `python src/train.py` before starting the dashboard.")
    st.stop()


TOPICS = [
    "teaching",
    "coursecontent",
    "examination",
    "labwork",
    "library",
    "extracurricular",
]


def analyze_texts(texts, topics=None):
    """Predict sentiment for a list of feedback comments."""
    cleaned = texts.fillna("").astype(str).str.strip()
    valid = cleaned.ne("")

    output = pd.DataFrame({
        "text": cleaned[valid].values
    })

    if topics is not None:
        output["topic"] = topics[valid].values

    if output.empty:
        return output

    output["sentiment"] = model.predict(output["text"].tolist())

    probabilities = model.predict_proba(output["text"].tolist())
    output["confidence"] = probabilities.max(axis=1)

    return output


def show_analysis(data):
    if data.empty:
        st.warning("No non-empty feedback comments to analyze.")
        return

    total = len(data)
    positive = int((data["sentiment"] == "positive").sum())
    negative = int((data["sentiment"] == "negative").sum())
    neutral = int((data["sentiment"] == "neutral").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Analyzed comments", total)
    col2.metric("Positive", positive)
    col3.metric("Neutral", neutral)
    col4.metric("Negative", negative)

    st.markdown("### Sentiment distribution")

    counts = (
        data["sentiment"]
        .value_counts()
        .rename_axis("Sentiment")
        .reset_index(name="Count")
    )

    fig = px.bar(
        counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        title="Predicted sentiment across feedback",
        color_discrete_map={
            "positive": "#16A34A",
            "neutral": "#64748B",
            "negative": "#DC2626",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    if "topic" in data.columns:
        st.markdown("### Feedback by topic")

        topic_counts = (
            data.groupby(["topic", "sentiment"])
            .size()
            .reset_index(name="Count")
        )

        topic_fig = px.bar(
            topic_counts,
            x="topic",
            y="Count",
            color="sentiment",
            barmode="group",
            title="Predicted sentiment by topic",
            color_discrete_map={
                "positive": "#16A34A",
                "neutral": "#64748B",
                "negative": "#DC2626",
            },
        )
        st.plotly_chart(topic_fig, use_container_width=True)

        st.markdown("### Issues for human review")
        st.subheader("Individual Predictions")
        
    # Complaint theme analysis based on negative predictions
    st.subheader("Complaint Theme Analysis")
    st.caption(
        "Themes are assigned using keyword rules. Review the original "
        "comments before treating a theme as a confirmed issue."
    )
    

    negative_comments = data[
        data["sentiment"].astype(str).str.lower().eq("negative")
    ].copy()

    if negative_comments.empty:
        st.info("No negative predictions are available for theme analysis.")
    else:
        negative_comments["matched_themes"] = (
            negative_comments["text"]
            .fillna("")
            .astype(str)
            .apply(identify_themes)
        )

        theme_rows = negative_comments.explode("matched_themes").copy()
        theme_rows["matched_themes"] = (
            theme_rows["matched_themes"].fillna("Other / Review")
        )

        theme_summary = (
            theme_rows["matched_themes"]
            .value_counts()
            .rename_axis("Theme")
            .reset_index(name="Matching comments")
        )

        total_negative = len(negative_comments)
        unmatched_count = int(
            negative_comments["matched_themes"].apply(
                lambda themes: "Other / Review" in themes
            ).sum()
        )

        metric1, metric2 = st.columns(2)
        metric1.metric("Negative comments", total_negative)
        metric2.metric("Comments needing review", unmatched_count)

        theme_fig = px.bar(
            theme_summary,
            x="Matching comments",
            y="Theme",
            orientation="h",
            title="Frequently Matched Complaint Themes",
            color="Matching comments",
            color_continuous_scale="Reds",
        )
        theme_fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(theme_fig, use_container_width=True)

        selected_theme = st.selectbox(
            "Inspect a complaint theme",
            options=theme_summary["Theme"].tolist(),
        )

        evidence = theme_rows[
            theme_rows["matched_themes"] == selected_theme
        ].copy()

        evidence_columns = ["text"]
        if "topic" in evidence.columns:
            evidence_columns.append("topic")
        
        
        st.markdown("### Suggested improvement action")
        st.info(get_recommendation(selected_theme))
        st.caption(
            "This is a rule-based suggestion. Review the original feedback "
            "and validate the issue before taking action."
        )


        st.markdown(f"**Original comments matching: {selected_theme}**")
        st.dataframe(
            evidence[evidence_columns],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Complaint Theme Evidence",
            data=theme_rows.to_csv(index=False).encode("utf-8"),
            file_name="campuspulse_complaint_themes.csv",
            mime="text/csv",
        )


        negative_data = data[data["sentiment"] == "negative"]

        if negative_data.empty:
            st.success("No negative predictions in this batch.")
        else:
            issues = (
                negative_data.groupby("topic")
                .size()
                .reset_index(name="Negative predictions")
                .sort_values(
                    "Negative predictions", ascending=False
                )
            )

            st.dataframe(issues, use_container_width=True)

            st.caption(
                "Rankings are based on predicted negative comments "
                "in the supplied batch. They indicate areas to review, "
                "not verified institutional problems."
            )

    st.markdown("### Individual predictions")

    display_data = data.copy()
    display_data["confidence"] = (
        display_data["confidence"] * 100
    ).round(1).astype(str) + "%"

    st.dataframe(display_data, use_container_width=True)

    csv_bytes = data.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "Download analyzed feedback (CSV)",
        data=csv_bytes,
        file_name="campuspulse_analysis.csv",
        mime="text/csv",
    
        
    )


if page == "Dashboard":
    st.markdown("## Feedback analysis dashboard")
    st.write(
        "Upload a CSV file to analyze multiple comments, "
        "or use the Analyze Feedback page to test one comment."
    )

    uploaded_file = st.file_uploader(
        "Upload feedback CSV",
        type=["csv"],
        key="dashboard_upload",
    )

    if uploaded_file is not None:
        try:
            raw_data = pd.read_csv(uploaded_file)

            if raw_data.empty or len(raw_data.columns) == 0:
                st.error("The uploaded CSV is empty.")
                st.stop()

            st.write("Preview of uploaded data")
            st.dataframe(raw_data.head(), use_container_width=True)

            text_column = st.selectbox(
                "Select the feedback text column",
                options=list(raw_data.columns),
            )

            topic_options = ["None"] + list(raw_data.columns)
            topic_column = st.selectbox(
                "Select a topic column (optional)",
                options=topic_options,
            )

            if st.button("Analyze uploaded feedback"):
                topics = None

                if topic_column != "None":
                    topics = raw_data[topic_column].astype(str)

                results = analyze_texts(
                    raw_data[text_column],
                    topics,
                )

                show_analysis(results)

        except Exception as error:
            st.error(f"Could not analyze the CSV: {error}")
    else:
        st.info(
            "Upload a CSV to see sentiment charts, topic insights, "
            "and downloadable predictions."
        )

elif page == "Analyze Feedback":
    st.markdown("## Analyze one student comment")

    feedback = st.text_area(
        "Enter student feedback",
        placeholder=(
            "Example: The teachers are helpful, but the lab "
            "equipment needs improvement."
        ),
        height=130,
    )

    if st.button("Analyze sentiment", type="primary"):
        if not feedback.strip():
            st.warning("Please enter feedback first.")
        else:
            try:
                result = predict_sentiment(feedback, model)

                sentiment = result["sentiment"]
                confidence = result["confidence"]

                st.markdown("### Prediction")

                if sentiment == "positive":
                    st.success("Predicted sentiment: Positive")
                elif sentiment == "negative":
                    st.error("Predicted sentiment: Negative")
                else:
                    st.info("Predicted sentiment: Neutral")

                st.metric(
                    "Model confidence",
                    f"{confidence:.1%}",
                )

                st.caption(
                    "Confidence is the model's estimated probability "
                    "for its chosen class, not a guarantee of correctness."
                )

                st.markdown("### Class probabilities")

                probability_data = pd.DataFrame({
                    "Sentiment": list(result["probabilities"].keys()),
                    "Probability": list(
                        result["probabilities"].values()
                    ),
                })

                fig = px.bar(
                    probability_data,
                    x="Sentiment",
                    y="Probability",
                    range_y=[0, 1],
                    title="Model class probabilities",
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as error:
                st.error(f"Prediction failed: {error}")

elif page == "About the Project":
    st.markdown("## About CampusPulse AI")

    st.write(
        "CampusPulse AI is a student feedback intelligence prototype "
        "that uses natural language processing and machine learning "
        "to classify feedback sentiment and help institutions identify "
        "areas that may need attention."
    )

    st.markdown("### Technology")
    st.write(
        "- Python and Pandas\n"
        "- TF-IDF text features\n"
        "- Logistic Regression\n"
        "- Scikit-learn\n"
        "- Streamlit and Plotly"
    )

    st.markdown("### Model evaluation")
    metrics_path = PROJECT_DIR / "reports" / "model_metrics.json"

    if metrics_path.exists():
        metrics = pd.read_json(metrics_path, typ="series")

        col1, col2 = st.columns(2)
        col1.metric(
            "Test accuracy",
            f"{float(metrics['accuracy']):.1%}",
        )
        col2.metric(
            "Macro F1",
            f"{float(metrics['macro_f1']):.4f}",
        )

        st.caption(
            "These results come from the saved evaluation report. "
            "They are not a guarantee of performance on new institutions."
        )
    else:
        st.info("Model evaluation report is not available yet.")
