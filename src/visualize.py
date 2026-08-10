"""
visualize.py
------------
Plotly chart builders used by dashboard.py. Kept separate from the
Streamlit app so the same figures can be reused in a notebook or exported
to static HTML (see `outputs/` via `python -m src.visualize`).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Brand-neutral, colorblind-safe categorical palette.
CATEGORICAL = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
               "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#EECA3B"]


def top_skills_bar(skill_freq_df, top_n=15):
    data = skill_freq_df.sort_values("count", ascending=True).tail(top_n)
    fig = px.bar(
        data, x="count", y="skill", orientation="h", color="category",
        color_discrete_sequence=CATEGORICAL,
        title=f"Top {top_n} In-Demand Skills Across All Postings",
        labels={"count": "# of postings mentioning skill", "skill": ""},
    )
    fig.update_layout(legend_title_text="Category", height=550)
    return fig


def skills_by_title_heatmap(df, top_titles=8, top_skills=15):
    top_title_list = df["title"].str.replace(r"^(Junior|Senior|Lead|Staff|Principal)\s+", "", regex=True)
    df = df.assign(base_title=top_title_list)
    top_title_names = df["base_title"].value_counts().head(top_titles).index.tolist()

    exploded = df[df["base_title"].isin(top_title_names)].explode("skills").dropna(subset=["skills"])
    top_skill_names = exploded["skills"].value_counts().head(top_skills).index.tolist()
    exploded = exploded[exploded["skills"].isin(top_skill_names)]

    pivot = (
        exploded.groupby(["base_title", "skills"]).size().unstack(fill_value=0)
    )
    pivot = pivot.reindex(index=top_title_names, columns=top_skill_names)
    pct = pivot.div(df[df["base_title"].isin(top_title_names)].groupby("base_title").size(), axis=0) * 100

    fig = go.Figure(data=go.Heatmap(
        z=pct.values, x=pct.columns, y=pct.index,
        colorscale="Blues", colorbar_title="% of postings",
    ))
    fig.update_layout(title="Skill Demand by Role (% of postings mentioning skill)", height=500)
    return fig


def salary_distribution_box(df):
    priced = df.dropna(subset=["salary_min", "salary_max"]).copy()
    priced["salary_mid"] = (priced["salary_min"] + priced["salary_max"]) / 2
    priced["base_title"] = priced["title"].str.replace(
        r"^(Junior|Senior|Lead|Staff|Principal)\s+", "", regex=True
    )
    order = priced.groupby("base_title")["salary_mid"].median().sort_values(ascending=False).index.tolist()
    fig = px.box(
        priced, x="base_title", y="salary_mid", color="base_title",
        category_orders={"base_title": order},
        color_discrete_sequence=CATEGORICAL,
        title="Salary Distribution by Role (USD, midpoint of posted range)",
        labels={"salary_mid": "Salary (USD)", "base_title": ""},
    )
    fig.update_layout(showlegend=False, height=500)
    return fig


def postings_by_platform_pie(df):
    counts = df["platform"].value_counts().reset_index()
    counts.columns = ["platform", "count"]
    fig = px.pie(
        counts, names="platform", values="count",
        color_discrete_sequence=CATEGORICAL,
        title="Postings by Platform",
        hole=0.4,
    )
    fig.update_layout(height=450)
    return fig


def postings_by_region_bar(df):
    def region_of(loc):
        if not isinstance(loc, str):
            return "Unknown"
        if "remote" in loc.lower():
            return "Remote"
        if any(c in loc for c in ["India"]):
            return "India"
        if any(c in loc for c in ["UK", "Germany", "Netherlands", "Ireland"]):
            return "Europe"
        if any(c in loc for c in ["Canada"]):
            return "Canada"
        if any(c in loc for c in ["Singapore", "Australia"]):
            return "APAC"
        return "US"

    df = df.copy()
    df["region"] = df["location"].apply(region_of)
    counts = df["region"].value_counts().reset_index()
    counts.columns = ["region", "count"]
    fig = px.bar(
        counts, x="region", y="count", color="region",
        color_discrete_sequence=CATEGORICAL,
        title="Postings by Region",
        labels={"count": "# of postings"},
    )
    fig.update_layout(showlegend=False, height=400)
    return fig


def topic_distribution_bar(topic_df, topic_summary):
    counts = topic_df["topic_label"].value_counts().reset_index()
    counts.columns = ["topic_label", "count"]
    fig = px.bar(
        counts, x="count", y="topic_label", orientation="h",
        color="topic_label", color_discrete_sequence=CATEGORICAL,
        title="Postings per Discovered Topic Cluster (LDA)",
        labels={"count": "# of postings", "topic_label": ""},
    )
    fig.update_layout(showlegend=False, height=400)
    return fig


def postings_over_time_line(df):
    dated = df.copy()
    dated["date_posted"] = pd.to_datetime(dated["date_posted"], errors="coerce")
    dated = dated.dropna(subset=["date_posted"])
    weekly = dated.set_index("date_posted").resample("W").size().reset_index(name="count")
    fig = px.line(
        weekly, x="date_posted", y="count",
        title="Postings Over Time (weekly)",
        labels={"date_posted": "", "count": "# of postings"},
    )
    fig.update_traces(line_color=CATEGORICAL[0])
    fig.update_layout(height=350)
    return fig


if __name__ == "__main__":
    import os
    from .text_analysis import add_skill_columns, skill_frequency
    from .topic_modeling import run_topic_modeling

    df = pd.read_csv("data/job_postings.csv")
    df = add_skill_columns(df)

    os.makedirs("outputs", exist_ok=True)
    top_skills_bar(skill_frequency(df)).write_html("outputs/top_skills.html")
    salary_distribution_box(df).write_html("outputs/salary_distribution.html")
    postings_by_platform_pie(df).write_html("outputs/postings_by_platform.html")
    postings_by_region_bar(df).write_html("outputs/postings_by_region.html")
    skills_by_title_heatmap(df).write_html("outputs/skills_by_title_heatmap.html")

    topic_df, topic_summary = run_topic_modeling(df)
    topic_distribution_bar(topic_df, topic_summary).write_html("outputs/topic_distribution.html")
    topic_summary.to_csv("outputs/topic_summary.csv", index=False)

    print("Wrote charts to outputs/")
