"""
dashboard.py
------------
Interactive Streamlit dashboard over the job market dataset. Run with:

    streamlit run dashboard.py

This is the "interactive dashboard" deliverable in place of a Tableau
workbook (Tableau itself isn't scriptable) - a "Export data for Tableau"
button on the Overview tab writes a clean, pre-aggregated CSV
(outputs/tableau_export.csv) shaped for a straight Tableau import if you
want to build a .twbx on top of it.
"""

import pandas as pd
import streamlit as st

from src.text_analysis import add_skill_columns, skill_frequency, salary_summary
from src.topic_modeling import run_topic_modeling
from src.resume_matcher import match_to_jd, match_to_market
from src.visualize import (
    top_skills_bar, skills_by_title_heatmap, salary_distribution_box,
    postings_by_platform_pie, postings_by_region_bar, topic_distribution_bar,
    postings_over_time_line,
)

st.set_page_config(page_title="Job Market Analytics · Data Science Roles", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("data/job_postings.csv")
    df = add_skill_columns(df)
    return df


@st.cache_data
def load_topics(df):
    return run_topic_modeling(df, n_topics=6)


df = load_data()
topic_df, topic_summary = load_topics(df)

st.title("Job Market Analytics for Data Science Roles")
st.caption(
    f"{len(df):,} postings analyzed "
    f"({(df['source_type'] == 'live_scraped').sum()} live-scraped from Remote OK / "
    f"We Work Remotely + {(df['source_type'] == 'synthetic').sum():,} synthetically "
    f"scaled for statistical volume — see README for methodology)."
)

tab_overview, tab_skills, tab_topics, tab_matcher = st.tabs(
    ["Overview", "Skill Demand", "Topic Clusters", "Resume Matcher"]
)

with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total postings", f"{len(df):,}")
    col2.metric("Unique companies", f"{df['company'].nunique():,}")
    col3.metric("Platforms", f"{df['platform'].nunique()}")
    priced = df.dropna(subset=["salary_min", "salary_max"])
    if len(priced):
        mid = ((priced["salary_min"] + priced["salary_max"]) / 2).median()
        col4.metric("Median salary (disclosed)", f"${mid:,.0f}")

    c1, c2 = st.columns(2)
    c1.plotly_chart(postings_by_platform_pie(df), width="stretch")
    c2.plotly_chart(postings_by_region_bar(df), width="stretch")
    st.plotly_chart(postings_over_time_line(df), width="stretch")
    st.plotly_chart(salary_distribution_box(df), width="stretch")

    if st.button("Export aggregated data for Tableau"):
        export = df.drop(columns=["skills"]).copy()
        export["skills_list"] = df["skills"].apply(lambda s: ", ".join(s))
        export.to_csv("outputs/tableau_export.csv", index=False)
        st.success("Wrote outputs/tableau_export.csv — ready to import into Tableau.")

with tab_skills:
    st.plotly_chart(top_skills_bar(skill_frequency(df)), width="stretch")
    st.plotly_chart(skills_by_title_heatmap(df), width="stretch")
    st.subheader("Median salary by role")
    st.dataframe(salary_summary(df), width="stretch")

with tab_topics:
    st.markdown(
        "LDA clusters postings into skill-based groups purely from description "
        "text, independent of the (noisy) `title` field."
    )
    st.plotly_chart(topic_distribution_bar(topic_df, topic_summary), width="stretch")
    st.subheader("Topic keywords")
    st.dataframe(topic_summary, width="stretch")

with tab_matcher:
    st.markdown("Compare a resume against a specific job description, or against overall market demand for a role.")
    mode = st.radio("Mode", ["Match against a JD", "Match against market demand"], horizontal=True)

    resume_text = st.text_area("Paste resume text", height=200, key="resume")

    if mode == "Match against a JD":
        jd_text = st.text_area("Paste job description", height=200, key="jd")
        if st.button("Run match") and resume_text and jd_text:
            with st.spinner("Scoring..."):
                result = match_to_jd(resume_text, jd_text)
            st.metric("Overall match score", f"{result['overall_score']}%")
            c1, c2 = st.columns(2)
            c1.metric("Semantic similarity", f"{result['semantic_similarity']}%")
            c2.metric("Skill coverage", f"{result['skill_coverage']}%")
            st.write("**Matched skills:**", ", ".join(result["matched_skills"]) or "none")
            st.write("**Missing skills:**", ", ".join(result["missing_skills"]) or "none")
    else:
        title_filter = st.text_input("Target role (matches `title` contains)", value="Data Scientist")
        region_filter = st.text_input("Region filter (optional, matches `location` contains)", value="")
        if st.button("Run market match") and resume_text:
            with st.spinner("Scoring..."):
                result = match_to_market(
                    resume_text, df,
                    title_contains=title_filter or None,
                    region=region_filter or None,
                )
            if result["n_postings_considered"] == 0:
                st.warning("No postings matched that filter.")
            else:
                st.metric("Market fit score", f"{result['market_fit_score']}%")
                st.caption(f"Based on {result['n_postings_considered']} matching postings.")
                have = ", ".join(f"{s['skill']} ({s['pct_of_postings']}%)" for s in result["have_skills"]) or "none"
                missing = ", ".join(f"{s['skill']} ({s['pct_of_postings']}%)" for s in result["missing_skills"]) or "none"
                st.write("**Skills you have that the market wants:**", have)
                st.write("**In-demand skills you're missing:**", missing)
