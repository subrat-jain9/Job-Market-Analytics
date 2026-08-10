"""
test_run.py
-----------
CLI smoke test for the full pipeline: load dataset -> skill extraction ->
TF-IDF/frequency analysis -> LDA topic modeling -> resume matcher (JD mode
and market mode). Run with: python test_run.py
"""

import pandas as pd

from src.text_analysis import add_skill_columns, skill_frequency, salary_summary
from src.topic_modeling import run_topic_modeling
from src.resume_matcher import match_to_jd, match_to_market

SAMPLE_RESUME = """
Data professional with 2 years of experience in Python, SQL, and pandas.
Built dashboards in Tableau and Power BI. Familiar with scikit-learn,
machine learning, and A/B testing. Worked with AWS and Docker.
"""

SAMPLE_JD = """
We're hiring a Data Scientist. Required: Python, SQL, machine learning,
scikit-learn, statistics. Nice to have: PyTorch, AWS, Airflow.
"""


def main():
    print("Loading dataset...")
    df = pd.read_csv("data/job_postings.csv")
    assert len(df) >= 5000, f"expected >=5000 postings, got {len(df)}"
    print(f"  {len(df)} postings loaded ({df['source_type'].value_counts().to_dict()})")

    print("Extracting skills...")
    df = add_skill_columns(df)
    assert df["n_skills"].sum() > 0
    print(f"  avg skills/posting: {df['n_skills'].mean():.1f}")

    print("Running skill frequency analysis...")
    freq = skill_frequency(df, top_n=10)
    print(freq[["skill", "count", "pct_of_postings"]].to_string(index=False))

    print("\nRunning salary summary...")
    print(salary_summary(df).head(5).to_string(index=False))

    print("\nRunning LDA topic modeling (this can take ~10-20s)...")
    topic_df, topic_summary = run_topic_modeling(df, n_topics=6)
    assert len(topic_summary) == 6
    print(topic_summary.to_string(index=False))

    print("\nRunning resume matcher (JD mode, TF-IDF)...")
    result = match_to_jd(SAMPLE_RESUME, SAMPLE_JD, method="tfidf")
    print(f"  overall={result['overall_score']}% semantic={result['semantic_similarity']}% "
          f"coverage={result['skill_coverage']}%")
    print(f"  matched={result['matched_skills']}")
    print(f"  missing={result['missing_skills']}")
    assert 0 <= result["overall_score"] <= 100

    print("\nRunning resume matcher (market mode)...")
    market_result = match_to_market(SAMPLE_RESUME, df, title_contains="Data Scientist")
    print(f"  postings considered: {market_result['n_postings_considered']}")
    print(f"  market fit score: {market_result['market_fit_score']}%")
    assert market_result["n_postings_considered"] > 0

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
