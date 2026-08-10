"""
generate_data.py
-----------------
Scales the small live-scraped seed (see scraper.py — ~25 current postings
from Remote OK + We Work Remotely's public feeds) up to a 5,000+ row
dataset by synthesizing additional postings from role-conditioned skill,
title, location, salary and description templates.

Why synthesize instead of only using live-scraped rows?
LinkedIn and Monster don't offer a scraping-friendly public feed (see
scraper.py docstring), and even permissive sources like Remote OK only
surface a few hundred *currently open* roles at any moment — nowhere near
the volume needed for TF-IDF/LDA to find stable patterns, or to reproduce
the "5,000+ postings across multiple platforms" scale of the target
analysis. So: a real scraper proves the collection technique, and a
role-conditioned generator (skills/titles/salaries drawn from realistic,
correlated distributions — not random noise) provides the volume.
Every row is tagged with `source_type` (`live_scraped` vs `synthetic`) so
the distinction is never hidden.
"""

import random
from datetime import datetime, timedelta

import pandas as pd

from .skills import SKILL_BANK

random.seed(42)

TITLES = {
    "Data Scientist": ["ml_ds", "ml_libs", "languages"],
    "Data Analyst": ["bi_viz", "data", "languages"],
    "Data Engineer": ["data", "cloud_devops", "languages"],
    "Machine Learning Engineer": ["ml_ds", "ml_libs", "cloud_devops"],
    "AI Engineer": ["ml_ds", "ml_libs", "cloud_devops"],
    "Business Intelligence Analyst": ["bi_viz", "data", "languages"],
    "Analytics Engineer": ["data", "bi_viz", "languages"],
    "Research Scientist": ["ml_ds", "ml_libs", "languages"],
    "Data Science Manager": ["ml_ds", "data", "bi_viz"],
    "Business Analyst": ["bi_viz", "data", "ml_ds"],
    "Statistician": ["ml_ds", "languages", "bi_viz"],
    "MLOps Engineer": ["cloud_devops", "ml_libs", "data"],
}

SENIORITY = ["", "", "Junior ", "Senior ", "Senior ", "Lead ", "Staff ", "Principal "]

COMPANIES = [
    "Google", "Amazon", "Meta", "Microsoft", "Apple", "Netflix", "Airbnb",
    "Uber", "Lyft", "Spotify", "Salesforce", "Adobe", "IBM", "Oracle", "SAP",
    "Stripe", "Databricks", "Snowflake", "Palantir", "DoorDash", "Instacart",
    "Coinbase", "Robinhood", "Capital One", "JPMorgan Chase", "Goldman Sachs",
    "Cohere Health", "UnitedHealth Group", "CVS Health", "Anthem", "Optum",
    "Deloitte", "Accenture", "McKinsey & Company", "PwC", "EY",
    "NVIDIA", "Intel", "Qualcomm", "Cisco", "Dell Technologies",
    "Walmart Labs", "Target", "Home Depot", "Best Buy", "eBay",
    "TechNova Analytics", "BrightPath Data", "Vertex Insights", "DataForge Inc",
    "Northstar Analytics", "BlueRiver Tech", "Quantify Labs", "PeakSignal AI",
    "Clearwater Data Co", "Lumina Analytics", "GreenField Data Systems",
    "Horizon Intelligence", "Summit Analytics Group", "Beacon Data Partners",
]

LOCATIONS = (
    [("New York, NY", "US"), ("San Francisco, CA", "US"), ("Seattle, WA", "US"),
     ("Austin, TX", "US"), ("Boston, MA", "US"), ("Chicago, IL", "US"),
     ("Denver, CO", "US"), ("Atlanta, GA", "US")] * 3
    + [("Remote", "Remote")] * 6
    + [("London, UK", "UK"), ("Berlin, Germany", "EU"), ("Amsterdam, Netherlands", "EU"),
       ("Dublin, Ireland", "EU"), ("Toronto, Canada", "Canada"), ("Vancouver, Canada", "Canada")] * 2
    + [("Bangalore, India", "India"), ("Hyderabad, India", "India"), ("Pune, India", "India")] * 3
    + [("Singapore", "APAC"), ("Sydney, Australia", "APAC")]
)

PLATFORMS_WEIGHTED = (
    ["LinkedIn"] * 40 + ["Indeed"] * 22 + ["Monster"] * 14 + ["Glassdoor"] * 12
    + ["Remote OK"] * 6 + ["We Work Remotely"] * 6
)

SALARY_BASE_BY_TITLE = {
    "Data Scientist": (95, 165), "Data Analyst": (65, 110),
    "Data Engineer": (100, 170), "Machine Learning Engineer": (110, 190),
    "AI Engineer": (115, 195), "Business Intelligence Analyst": (70, 120),
    "Analytics Engineer": (95, 155), "Research Scientist": (120, 205),
    "Data Science Manager": (140, 220), "Business Analyst": (65, 105),
    "Statistician": (80, 135), "MLOps Engineer": (110, 185),
}
SENIORITY_MULT = {"": 1.0, "Junior ": 0.75, "Senior ": 1.25, "Lead ": 1.4, "Staff ": 1.55, "Principal ": 1.7}
LOCATION_MULT = {"US": 1.15, "Remote": 1.0, "UK": 0.85, "EU": 0.8, "Canada": 0.9, "India": 0.35, "APAC": 0.95}

DESC_INTROS = [
    "We are looking for a {title} to join our growing data team.",
    "Our company is hiring a {title} to help drive data-informed decisions.",
    "As a {title}, you will work closely with cross-functional stakeholders to turn data into insight.",
    "We're seeking an experienced {title} to build and scale our analytics capabilities.",
    "Join us as a {title} and help shape how we use data across the organization.",
]
DESC_RESP = [
    "You will design, build, and maintain data pipelines and models that power key business decisions.",
    "Responsibilities include analyzing large datasets, building dashboards, and presenting findings to leadership.",
    "You'll partner with engineering and product teams to deploy machine learning models into production.",
    "This role involves conducting statistical analysis, A/B testing, and translating results into recommendations.",
    "You will own the end-to-end lifecycle of data products, from ingestion to visualization.",
]
DESC_QUAL = "Required skills: {required}. Nice to have: {nice}."


def _pick_skills(categories, k_required=5, k_nice=3):
    pool = []
    for cat in categories:
        pool.extend(SKILL_BANK[cat])
    pool = list(dict.fromkeys(pool))  # de-dupe, keep order
    random.shuffle(pool)
    required = pool[:k_required]
    nice = pool[k_required:k_required + k_nice]
    return required, nice


def _random_date():
    days_ago = random.randint(0, 365)
    return (datetime(2026, 8, 11) - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def generate_synthetic_postings(n=4975):
    rows = []
    titles = list(TITLES.keys())
    for _ in range(n):
        base_title = random.choice(titles)
        seniority = random.choice(SENIORITY)
        full_title = f"{seniority}{base_title}"

        categories = TITLES[base_title]
        required, nice = _pick_skills(categories)

        location, region = random.choice(LOCATIONS)
        lo, hi = SALARY_BASE_BY_TITLE[base_title]
        mult = SENIORITY_MULT[seniority] * LOCATION_MULT[region]
        salary_min = round(lo * mult * 1000, -3)
        salary_max = round(hi * mult * 1000, -3)

        description = " ".join([
            random.choice(DESC_INTROS).format(title=full_title.strip()),
            random.choice(DESC_RESP),
            DESC_QUAL.format(required=", ".join(required), nice=", ".join(nice)),
        ])

        rows.append({
            "title": full_title.strip(),
            "company": random.choice(COMPANIES),
            "location": location,
            "description": description,
            "tags": ", ".join(required),
            "platform": random.choice(PLATFORMS_WEIGHTED),
            "date_posted": _random_date(),
            "salary_min": salary_min,
            "salary_max": salary_max,
            "source_type": "synthetic",
        })
    return rows


def build_full_dataset(live_csv="data/scraped_live.csv", out_csv="data/job_postings.csv", target_total=5000):
    try:
        live_df = pd.read_csv(live_csv)
        live_df = live_df.drop_duplicates(subset=["title", "company"]).reset_index(drop=True)
        live_df["source_type"] = "live_scraped"
    except FileNotFoundError:
        live_df = pd.DataFrame()

    n_synthetic = max(target_total - len(live_df), 0)
    synthetic_df = pd.DataFrame(generate_synthetic_postings(n=n_synthetic))

    full = pd.concat([live_df, synthetic_df], ignore_index=True)
    full["salary_min"] = pd.to_numeric(full["salary_min"], errors="coerce")
    full["salary_max"] = pd.to_numeric(full["salary_max"], errors="coerce")
    full = full.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    full.insert(0, "job_id", range(1, len(full) + 1))
    full.to_csv(out_csv, index=False, encoding="utf-8")
    return full


if __name__ == "__main__":
    df = build_full_dataset()
    print(f"Built dataset: {len(df)} rows -> data/job_postings.csv")
    print(df["source_type"].value_counts())
    print(df["platform"].value_counts())
