"""
resume_matcher.py
------------------
Resume-to-job-market matcher, reusing the scoring architecture from this
user's Smart Resume Agent project:

    overall = 60% semantic_similarity + 40% skill_coverage

Two modes:
  1. match_to_jd(resume, jd)            - classic one-JD matcher.
  2. match_to_market(resume, df, ...)   - compares a resume against the
     *aggregate* in-demand skills for a role/region slice of the scraped
     job market dataset, so a candidate can see "of the skills the market
     actually asks for in Data Scientist roles, which do you have / miss,"
     ranked by how many postings mention each one.
"""

from .skills import find_skills, normalise


def _cosine(a, b):
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / denom)))


_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _semantic_similarity(text_a, text_b, method="embeddings"):
    if method == "embeddings":
        try:
            model = _load_model()
            vectors = model.encode([text_a, text_b])
            return _cosine(vectors[0], vectors[1]), "embeddings"
        except Exception:
            method = "tfidf"
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform([text_a, text_b])
    return _cosine(matrix[0].toarray()[0], matrix[1].toarray()[0]), "tfidf (fallback)"


def match_to_jd(resume, jd, method="embeddings"):
    """Classic single-JD match: semantic similarity + skill coverage."""
    semantic, used = _semantic_similarity(resume, jd, method=method)

    resume_skills = normalise(find_skills(resume))
    jd_skills = normalise(find_skills(jd))
    matched = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)
    coverage = (len(matched) / len(jd_skills)) if jd_skills else 0.0

    overall = 0.6 * semantic + 0.4 * coverage
    return {
        "overall_score": round(overall * 100),
        "semantic_similarity": round(semantic * 100),
        "skill_coverage": round(coverage * 100),
        "method_used": used,
        "matched_skills": matched,
        "missing_skills": missing,
        "jd_skills": sorted(jd_skills),
        "resume_skills": sorted(resume_skills),
    }


def market_demand_skills(df, title_contains=None, region=None, top_n=15):
    """
    Rank skills by % of matching postings that mention them - the
    "what does the market actually want" signal, independent of any one JD.
    `df` must already have a `skills` column (see text_analysis.add_skill_columns).
    """
    sub = df
    if title_contains:
        sub = sub[sub["title"].str.contains(title_contains, case=False, na=False)]
    if region:
        sub = sub[sub["location"].str.contains(region, case=False, na=False)]

    n_postings = len(sub)
    if n_postings == 0:
        return [], 0

    counts = {}
    for skills in sub["skills"]:
        for s in skills:
            counts[s] = counts.get(s, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    ranked_pct = [(skill, round(100 * count / n_postings, 1)) for skill, count in ranked]
    return ranked_pct, n_postings


def match_to_market(resume, df, title_contains=None, region=None, top_n=15):
    """
    Compare a resume's skills against the top-N most in-demand skills for a
    role/region slice of the job market dataset.
    """
    resume_skills = normalise(find_skills(resume))
    demand, n_postings = market_demand_skills(df, title_contains, region, top_n)

    have, missing = [], []
    for skill, pct in demand:
        target = have if skill in resume_skills else missing
        target.append({"skill": skill, "pct_of_postings": pct})

    coverage = (len(have) / len(demand)) if demand else 0.0
    return {
        "n_postings_considered": n_postings,
        "market_fit_score": round(coverage * 100),
        "have_skills": have,
        "missing_skills": missing,
        "resume_skills": sorted(resume_skills),
    }
