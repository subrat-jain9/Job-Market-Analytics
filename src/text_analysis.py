"""
text_analysis.py
-----------------
Keyword frequency + TF-IDF analysis of job posting descriptions:
  - Which skills/tools appear most often overall, and by region/title/platform.
  - TF-IDF to surface the terms that most distinguish one role/region from
    the rest of the corpus (frequency alone over-weights generic words like
    "data" or "team"; TF-IDF down-weights terms common across every posting).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .skills import find_skills, normalise, SKILL_TO_CATEGORY


def add_skill_columns(df):
    """Attach a `skills` column (sorted list of detected bank skills) to df."""
    df = df.copy()
    df["skills"] = df["description"].fillna("").apply(
        lambda text: sorted(normalise(find_skills(text)))
    )
    df["n_skills"] = df["skills"].apply(len)
    return df


def skill_frequency(df, group_col=None, top_n=20):
    """
    Overall (or per-group) skill mention counts, as a tidy DataFrame:
    columns = [group_col?, skill, category, count, pct_of_postings]
    """
    rows = []
    if group_col is None:
        groups = [(None, df)]
    else:
        groups = df.groupby(group_col)

    for group_value, sub in groups:
        n_postings = len(sub)
        counts = {}
        for skills in sub["skills"]:
            for s in skills:
                counts[s] = counts.get(s, 0) + 1
        for skill, count in counts.items():
            row = {
                "skill": skill,
                "category": SKILL_TO_CATEGORY.get(skill, "other"),
                "count": count,
                "pct_of_postings": round(100 * count / n_postings, 1) if n_postings else 0,
            }
            if group_col is not None:
                row[group_col] = group_value
            rows.append(row)

    result = pd.DataFrame(rows)
    sort_cols = [group_col, "count"] if group_col else ["count"]
    result = result.sort_values(sort_cols, ascending=[True, False] if group_col else False)
    if top_n and group_col is None:
        result = result.head(top_n)
    elif top_n and group_col:
        result = result.groupby(group_col, group_keys=False).apply(
            lambda g: g.head(top_n), include_groups=False
        ).reset_index()
    return result.reset_index(drop=True)


def top_tfidf_terms_by_group(df, group_col, top_n=15, max_features=3000):
    """
    For each value of `group_col` (e.g. title or region), find the terms
    whose TF-IDF score is highest *for that group's postings on average* —
    i.e. the vocabulary that most distinguishes that group from the rest
    of the corpus.
    """
    corpus = df["description"].fillna("").tolist()
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=max_features, ngram_range=(1, 2),
        min_df=5, max_df=0.4,
    )
    matrix = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()

    tfidf_df = pd.DataFrame(matrix.toarray(), columns=terms)
    tfidf_df[group_col] = df[group_col].values

    results = {}
    for group_value, sub in tfidf_df.groupby(group_col):
        mean_scores = sub.drop(columns=[group_col]).mean().sort_values(ascending=False)
        results[group_value] = mean_scores.head(top_n)
    return results


def salary_summary(df, group_col="title"):
    """Median/mean salary range by group, for postings that disclose salary."""
    priced = df.dropna(subset=["salary_min", "salary_max"]).copy()
    priced["salary_mid"] = (priced["salary_min"] + priced["salary_max"]) / 2
    summary = (
        priced.groupby(group_col)["salary_mid"]
        .agg(median_salary="median", mean_salary="mean", n_postings="count")
        .sort_values("median_salary", ascending=False)
        .reset_index()
    )
    return summary
