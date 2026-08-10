# Architecture (deep dive)

This is the engineer-facing companion to the two overview diagrams in
[README.md](README.md). Where those show *what* happens at a glance, these
show *how* — actual function call order, library calls, and how the
DataFrame's schema grows as it moves through the pipeline.

---

## 1. Data schema evolution

The same DataFrame gets columns bolted on as it passes through each stage.
This is the shape to have in your head before reading the sequence diagrams
below.

```mermaid
flowchart LR
    A["scraper.py output<br/>title, company, location,<br/>description, tags, platform,<br/>date_posted, salary_min, salary_max"]
    A --> B["+ source_type<br/>(generate_data.build_full_dataset)"]
    B --> C["+ job_id<br/>(after concat + shuffle)"]
    C --> D["job_postings.csv<br/>= the on-disk contract<br/>every downstream module reads"]
    D --> E["+ skills, n_skills<br/>(text_analysis.add_skill_columns)"]
    E --> F["+ topic_id, topic_label,<br/>topic_confidence<br/>(topic_modeling.assign_dominant_topic)"]
    F --> G["dashboard.py holds this<br/>fully-enriched frame in<br/>st.cache_data for the whole session"]
```

`job_postings.csv` itself never contains `skills`/`topic_*` columns — those
are computed at load time (`add_skill_columns`, `run_topic_modeling`) by
whichever script needs them, not persisted. If you're grepping the CSV for
a skill, you won't find it as a column.

---

## 2. Data collection: `python -m src.scraper` then `python -m src.generate_data`

```mermaid
sequenceDiagram
    participant CLI as caller (CLI / build_full_dataset)
    participant Scraper as scraper.py
    participant RemoteOK as Remote OK JSON API
    participant WWR as We Work Remotely RSS
    participant Gen as generate_data.py
    participant CSV as job_postings.csv

    CLI->>Scraper: scrape_all(remoteok_limit=600)
    Scraper->>RemoteOK: GET remoteok.com/api
    RemoteOK-->>Scraper: JSON array (index 0 = legal notice, rest = jobs)
    loop each job after index 0
        Scraper->>Scraper: _is_ds_relevant(title, tags, description)
        Note right of Scraper: regex over data science /<br/>ML / analytics / BI keywords
    end
    Scraper->>WWR: GET /categories/{cat}.rss<br/>(data-science, programming, product)
    WWR-->>Scraper: RSS/XML per category
    Scraper->>Scraper: BeautifulSoup(resp.content, "xml")<br/>parse <item> -> title/company/description
    Scraper->>Scraper: _is_ds_relevant() filter again
    Scraper-->>CSV: scraped_live.csv (~20-30 rows, source_type unset yet)

    CLI->>Gen: build_full_dataset(live_csv, out_csv, target_total=5000)
    Gen->>CSV: pd.read_csv(scraped_live.csv)
    Gen->>Gen: drop_duplicates(subset=[title, company])
    Gen->>Gen: live_df["source_type"] = "live_scraped"
    Gen->>Gen: n_synthetic = 5000 - len(live_df)
    Gen->>Gen: generate_synthetic_postings(n=n_synthetic)
    loop for each of n_synthetic rows (random.seed(42))
        Gen->>Gen: title = random(TITLES) + random(SENIORITY)
        Gen->>Gen: _pick_skills(TITLES[title]) -> required[5], nice[3]<br/>(shuffled skill-bank categories mapped to that title)
        Gen->>Gen: location = random(LOCATIONS) (weighted list, not uniform)
        Gen->>Gen: salary = SALARY_BASE_BY_TITLE[title]<br/>x SENIORITY_MULT[seniority] x LOCATION_MULT[region]
        Gen->>Gen: description = intro + responsibility + "Required: ...Nice to have: ..."<br/>(sampled from small template pools)
    end
    Gen->>Gen: pd.concat([live_df, synthetic_df])
    Gen->>Gen: df.sample(frac=1, random_state=42) — shuffle so live rows<br/>aren't all clustered at the top
    Gen->>Gen: df.insert(0, "job_id", range(1, len(df)+1))
    Gen-->>CSV: job_postings.csv (5,000 rows, encoding="utf-8")
```

**Why this matters if you're reading the code:** `scrape_all()` wraps each
source in its own `try/except requests.RequestException` — one source being
down (rate-limited, DNS hiccup) never blocks the other, and never blocks
`generate_data.py`, since `build_full_dataset` tolerates a missing
`scraped_live.csv` entirely (`except FileNotFoundError: live_df = pd.DataFrame()`)
and just synthesizes the full 5,000 rows.

---

## 3. Skill extraction + keyword/TF-IDF analysis

```mermaid
sequenceDiagram
    participant Caller as dashboard.py / test_run.py
    participant TA as text_analysis.py
    participant SK as skills.py
    participant TFIDF as sklearn.TfidfVectorizer

    Caller->>TA: add_skill_columns(df)
    loop each row's description
        TA->>SK: find_skills(text)
        SK->>SK: lowercase text, run all 120 precompiled<br/>regex patterns (lookaround-guarded, not \b)
        SK-->>TA: raw matched skill set
        TA->>SK: normalise(skills)
        SK->>SK: collapse aliases via ALIASES dict<br/>(sklearn -> scikit-learn, golang -> go, ...)
        SK-->>TA: canonical skill set
    end
    TA-->>Caller: df with [skills: list[str], n_skills: int]

    Caller->>TA: skill_frequency(df, group_col="title", top_n=15)
    TA->>TA: for each group, Counter over exploded skills list
    TA->>TA: pct_of_postings = 100 x count / len(group)
    TA-->>Caller: tidy frame [group_col?, skill, category, count, pct_of_postings]

    Caller->>TA: top_tfidf_terms_by_group(df, group_col)
    TA->>TFIDF: fit_transform(df.description)<br/>ngram_range=(1,2), min_df=5, max_df=0.4
    TFIDF-->>TA: sparse doc x term matrix
    TA->>TA: attach group_col, groupby().mean() per term
    TA-->>Caller: {group_value: top-N terms by mean TF-IDF weight}
```

`find_skills` is a **lookup against a fixed ~120-term vocabulary**, not a
free-form extractor (no spaCy NER, no LLM call) — every skill it reports is
literally present, character-for-character (case-insensitive), in the
posting text. That's a deliberate trade-off: it can't discover a skill
outside the bank, but it can never hallucinate one either, and the
match/miss lists in the resume matcher are always independently verifiable
against the source text.

---

## 4. LDA topic modeling

```mermaid
sequenceDiagram
    participant Caller as dashboard.py / test_run.py
    participant TM as topic_modeling.py
    participant CV as sklearn.CountVectorizer
    participant LDA as sklearn.LatentDirichletAllocation

    Caller->>TM: run_topic_modeling(df, n_topics=6)
    TM->>CV: fit_transform(df.description)<br/>ngram_range=(1,2), min_df=5, max_df=0.4
    Note right of CV: raw counts, not TF-IDF — LDA is a<br/>generative model over word counts
    CV-->>TM: doc-term count matrix
    TM->>LDA: fit_transform(doc_term_matrix)<br/>n_components=6, learning_method="online",<br/>max_iter=15, random_state=42
    LDA-->>TM: doc_topic_matrix (n_docs x 6 probabilities, rows sum to 1)
    TM->>TM: top_words_per_topic():<br/>argsort each topic's word-weight vector, take top 10
    TM->>TM: label_topics():<br/>join each topic's top 3 words -> human-readable label
    TM->>TM: assign_dominant_topic():<br/>topic_id = argmax(doc_topic_matrix, axis=1)<br/>topic_confidence = max probability
    TM-->>Caller: (df + [topic_id, topic_label, topic_confidence],<br/>topic_summary DataFrame)
```

`max_df=0.4` is load-bearing here, not cosmetic: without it, phrases that
appear in nearly every synthetic posting ("required skills", "nice to
have") dominate every topic and the clusters collapse into noise. Dropping
terms that appear in >40% of documents forces LDA to cluster on what
actually varies between postings.

---

## 5. Resume matcher — two call paths

```mermaid
sequenceDiagram
    participant UI as dashboard.py (Resume Matcher tab)
    participant RM as resume_matcher.py
    participant ST as SentenceTransformer
    participant SK as skills.py

    rect rgb(235, 245, 255)
    Note over UI,SK: Mode A — Match against a JD
    UI->>RM: match_to_jd(resume, jd, method="embeddings")
    RM->>RM: _semantic_similarity(resume, jd)
    RM->>ST: _load_model() -> SentenceTransformer("all-MiniLM-L6-v2")
    Note right of ST: module-level _MODEL cache —<br/>loaded once per process, reused after
    alt model available
        ST-->>RM: encode([resume, jd]) -> cosine similarity
    else load raises (offline, no local cache)
        RM->>RM: except Exception -> TfidfVectorizer cosine similarity<br/>method_used = "tfidf (fallback)"
    end
    RM->>SK: find_skills(resume), find_skills(jd) -> normalise() each
    RM->>RM: coverage = |matched| / |jd_skills| (0 if jd has no bank skills)
    RM->>RM: overall_score = round(100 x (0.6*semantic + 0.4*coverage))
    RM-->>UI: {overall_score, semantic_similarity, skill_coverage,<br/>matched_skills, missing_skills, method_used}
    end

    rect rgb(255, 245, 235)
    Note over UI,SK: Mode B — Match against market demand
    UI->>RM: match_to_market(resume, df, title_contains, region)
    RM->>RM: market_demand_skills(df, title_contains, region, top_n=15)
    RM->>RM: sub = df filtered by title.str.contains() and/or<br/>location.str.contains() (case-insensitive)
    RM->>RM: Counter over sub["skills"], rank desc, top 15,<br/>express as % of matching postings
    RM->>SK: find_skills(resume) -> normalise()
    RM->>RM: for each of the 15 demanded skills:<br/>resume has it -> have[], else -> missing[]
    RM->>RM: market_fit_score = round(100 x |have| / 15)
    RM-->>UI: {market_fit_score, have_skills, missing_skills,<br/>n_postings_considered, resume_skills}
    end
```

Both modes share the same `0.6 semantic + 0.4 coverage` shape conceptually,
but Mode B has no semantic term at all — "market fit" is purely
skill-coverage against a *ranked, weighted* list (each skill's weight is
"% of postings asking for it"), not a binary in/out list. That's the actual
difference between "how well do I match this one JD" and "how well do I
match what the market wants" — the second one cares about *how common* each
missing skill is, not just whether it's missing.

---

## 6. Dashboard wiring

```mermaid
flowchart TD
    Start(["streamlit run dashboard.py"]) --> LD["load_data()<br/>@st.cache_data<br/>pd.read_csv + add_skill_columns"]
    LD --> LT["load_topics(df)<br/>@st.cache_data<br/>run_topic_modeling(df, n_topics=6)"]
    LT --> Tabs{"st.tabs()"}

    Tabs --> T1["Overview"]
    T1 --> T1a["st.metric x4:<br/>postings, companies, platforms, median salary"]
    T1 --> T1b["postings_by_platform_pie, postings_by_region_bar,<br/>postings_over_time_line, salary_distribution_box"]
    T1 --> T1c["Export button -> outputs/tableau_export.csv"]

    Tabs --> T2["Skill Demand"]
    T2 --> T2a["top_skills_bar(skill_frequency(df))"]
    T2 --> T2b["skills_by_title_heatmap(df)"]
    T2 --> T2c["st.dataframe(salary_summary(df))"]

    Tabs --> T3["Topic Clusters"]
    T3 --> T3a["topic_distribution_bar(topic_df, topic_summary)"]
    T3 --> T3b["st.dataframe(topic_summary)"]

    Tabs --> T4["Resume Matcher"]
    T4 --> T4Radio{"st.radio: mode"}
    T4Radio -->|"JD mode"| M1["match_to_jd(resume, jd)"]
    T4Radio -->|"Market mode"| M2["match_to_market(resume, df, title, region)"]
```

Both `load_data()` and `load_topics()` are `@st.cache_data` — on a normal
session, the CSV is read once and LDA is fit once, no matter how many times
you switch tabs or re-run widgets. The two resume-matcher calls (`M1`/`M2`)
are the only functions on this page invoked on-demand (inside `st.button`
handlers), since they're the only ones with per-user input.

---

## Related reading

- [README.md](README.md) — plain-English and high-level technical diagrams,
  the honest write-up of what's live-scraped vs. synthetic, and how to run
  everything locally.
- Each `src/*.py` module has its own docstring explaining *why* it's built
  the way it is, not just what it does — worth reading directly if you want
  more than this file covers.
