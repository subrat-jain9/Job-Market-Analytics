"""
topic_modeling.py
------------------
Latent Dirichlet Allocation (LDA) over job posting descriptions, to
discover skill-based role clusters (e.g. "modeling/stats", "data
engineering/pipelines", "BI/reporting") without relying on the free-text
`title` field, which is noisy (a "Data Analyst" at one company can be a
"Business Analyst" doing the same job at another).

Pipeline: CountVectorizer (LDA wants raw term counts, not TF-IDF weights)
-> LatentDirichletAllocation -> for each posting, the dominant topic ID
becomes a `topic` column that downstream code (dashboard, resume matcher)
can use as a normalized "role category."
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def fit_topics(df, n_topics=6, max_features=2000, random_state=42):
    """
    Fit an LDA model on df["description"]. Returns
    (doc_topic_matrix, topic_term_matrix, vectorizer, model).
    """
    vectorizer = CountVectorizer(
        stop_words="english", max_features=max_features, ngram_range=(1, 2),
        min_df=5, max_df=0.4,
        # max_df drops boilerplate phrases that show up in most postings
        # regardless of role (e.g. "required skills", "nice to have") so
        # LDA clusters on what actually varies between postings.
    )
    doc_term_matrix = vectorizer.fit_transform(df["description"].fillna(""))

    model = LatentDirichletAllocation(
        n_components=n_topics, random_state=random_state, learning_method="online",
        max_iter=15,
    )
    doc_topic_matrix = model.fit_transform(doc_term_matrix)

    return doc_topic_matrix, vectorizer, model


def top_words_per_topic(vectorizer, model, top_n=10):
    """Return {topic_id: [top terms]} for labeling each topic."""
    terms = vectorizer.get_feature_names_out()
    topics = {}
    for topic_id, component in enumerate(model.components_):
        top_indices = component.argsort()[::-1][:top_n]
        topics[topic_id] = [terms[i] for i in top_indices]
    return topics


def label_topics(topic_words):
    """
    Turn each topic's top words into a short human-readable label by
    joining its 3 most distinctive terms — good enough to tell topics
    apart in a legend without hand-curated names.
    """
    return {
        topic_id: " / ".join(words[:3])
        for topic_id, words in topic_words.items()
    }


def assign_dominant_topic(df, doc_topic_matrix, topic_labels):
    """Attach `topic_id`, `topic_label`, and `topic_confidence` to df."""
    df = df.copy()
    dominant = doc_topic_matrix.argmax(axis=1)
    confidence = doc_topic_matrix.max(axis=1)
    df["topic_id"] = dominant
    df["topic_label"] = [topic_labels[t] for t in dominant]
    df["topic_confidence"] = np.round(confidence, 3)
    return df


def run_topic_modeling(df, n_topics=6):
    """Convenience wrapper: fit LDA, label topics, attach to df."""
    doc_topic_matrix, vectorizer, model = fit_topics(df, n_topics=n_topics)
    topic_words = top_words_per_topic(vectorizer, model)
    topic_labels = label_topics(topic_words)
    df_with_topics = assign_dominant_topic(df, doc_topic_matrix, topic_labels)
    topic_summary = pd.DataFrame([
        {"topic_id": tid, "label": topic_labels[tid], "top_words": ", ".join(words)}
        for tid, words in topic_words.items()
    ])
    return df_with_topics, topic_summary
