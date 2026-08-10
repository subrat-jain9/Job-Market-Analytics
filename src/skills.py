"""
skills.py
---------
Curated bank of skills/tools/titles commonly found in Data Science job
postings, plus helpers to detect which of them appear in a piece of text.

Reused from the Smart Resume Agent project (same "curated bank, not
free-form NER" design: keeps every match explainable and human-verifiable),
extended with analytics-specific tools (Tableau, Power BI, Excel, stats)
that matter more for DS/analyst postings than for generic SWE postings.
"""

import re

SKILL_BANK = {
    "languages": [
        "python", "java", "c++", "c#", "javascript", "typescript",
        "go", "golang", "rust", "r", "scala", "sql", "bash", "matlab",
    ],
    "ml_ds": [
        "machine learning", "deep learning", "neural networks", "nlp",
        "natural language processing", "computer vision", "reinforcement learning",
        "data science", "data analysis", "feature engineering", "model deployment",
        "supervised learning", "unsupervised learning", "classification",
        "regression", "clustering", "recommendation systems", "time series",
        "generative ai", "large language models", "llm", "llms", "rag",
        "prompt engineering", "fine-tuning", "embeddings", "transformers",
        "statistics", "statistical analysis", "a/b testing", "hypothesis testing",
        "experiment design", "forecasting", "anomaly detection",
    ],
    "ml_libs": [
        "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch",
        "numpy", "pandas", "matplotlib", "seaborn", "scipy", "xgboost",
        "lightgbm", "hugging face", "huggingface", "spacy", "nltk",
        "opencv", "langchain", "openai", "streamlit", "gradio", "plotly",
    ],
    "data": [
        "mysql", "postgresql", "postgres", "mongodb", "sqlite", "redis",
        "spark", "pyspark", "hadoop", "kafka", "airflow", "etl", "elt",
        "data pipeline", "data warehouse", "bigquery", "snowflake",
        "redshift", "dbt", "databricks",
    ],
    "bi_viz": [
        "tableau", "power bi", "looker", "excel", "google sheets",
        "d3.js", "qlik", "superset",
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "ci/cd", "github actions", "jenkins", "terraform", "linux",
        "rest api", "fastapi", "flask", "django", "microservices",
        "mlops", "model serving", "s3", "ec2", "lambda",
    ],
    "tools": [
        "git", "github", "gitlab", "jira", "vs code", "jupyter",
        "unit testing", "agile", "scrum",
    ],
}


def _flatten_bank():
    all_skills = set()
    for group in SKILL_BANK.values():
        for skill in group:
            all_skills.add(skill.lower())
    return all_skills


ALL_SKILLS = _flatten_bank()

# Which bucket each skill belongs to, e.g. "python" -> "languages".
SKILL_TO_CATEGORY = {
    skill.lower(): category
    for category, skills in SKILL_BANK.items()
    for skill in skills
}


def _skill_pattern(skill):
    """
    Match `skill` as a standalone token/phrase. Word boundaries (\\b) don't
    work for skills containing +, #, . or / (c++, c#, ci/cd), so we guard
    with lookarounds instead.
    """
    escaped = re.escape(skill)
    return re.compile(r"(?<![a-z0-9+#./])" + escaped + r"(?![a-z0-9+#./])")


_PATTERNS = {skill: _skill_pattern(skill) for skill in ALL_SKILLS}


def find_skills(text):
    """Return the set of bank skills that appear in `text` (case-insensitive)."""
    if not text:
        return set()
    lowered = text.lower()
    found = set()
    for skill, pattern in _PATTERNS.items():
        if pattern.search(lowered):
            found.add(skill)
    return found


ALIASES = {
    "sklearn": "scikit-learn",
    "golang": "go",
    "postgres": "postgresql",
    "huggingface": "hugging face",
    "llms": "llm",
    "natural language processing": "nlp",
    "google cloud": "gcp",
    "pyspark": "spark",
}


def normalise(skills):
    """Collapse alias skills onto their canonical name."""
    return {ALIASES.get(s, s) for s in skills}
